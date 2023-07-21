from collections import OrderedDict
from copy import deepcopy
import numpy as np
import logging
import time
from icecream import ic
from os.path import join

from mujoco_py import load_model_from_xml, MjSim
from autolab_core import RigidTransform
from hydra.utils import to_absolute_path
import klampt
from klampt import vis
from klampt.plan import robotplanning
from klampt import IKObjective, IKSolver
from klampt.model import ik
from stat_utils import sample_from_distrib as sample
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.objects import DoorObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from recovery_skills.graph.state import State
import robosuite.utils.transform_utils as T
from robosuite.utils.observables import create_gaussian_noise_corrupter

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")


class FrankaDoorEnv(SingleArmEnv):
    # needed to reset a state from a pillar state
    # uniquely defines a simulation state
    # more properly called world_state_vars
    state_vars = [
        "robot:arm/joints",
        "door:pose/position",
        "door:pose/theta",
        "hinge:pose/theta",
        'handle:pose/position',
        "handle:pose/theta",
    ]

    # state_var_ndims = [9, 3, 1, 1, 1]
    state_var_ndims = [9, 3, 1, 1, 3, 1]

    # To be used to compute distance
    rl_state_vars = [
        "robot_eef:pose/position",
        "robot_eef:pose/quat",
        "robot_eef:gripper/position",
        "hinge:pose/theta",
        "handle:pose/position",
        "handle:pose/theta",
        # "handle_center_of_rotation:pose/position",
    ]

    rl_state_var_ndims = [3, 4, 1, 1, 3, 1] #, 3]
    # stays constant
    # TODO Add handle type (lever, knob etc) and door type (door, cabinet)
    context_vars = ["door:pose/position",
                    "door:pose/theta",
                    "handle:dims"]

    context_var_ndims = [
        3,
        1,
        3
    ]


    """
    This class corresponds to the door opening task for a single robot arm.

    Args:
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        use_latch (bool): if True, uses a spring-loaded handle and latch to "lock" the door closed initially
            Otherwise, door is instantiated with a fixed handle

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        controller_configs=None,
        has_renderer=False,
        use_latch=True,
        reward_scale=1.0,
        placement_initializer=None,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_visual_mesh=True,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        context_cfg=None,
        obs_uncertainty=None,
        timestep=0.002,
        eef_start_region_cfg=None,
    ):
        self.table_full_size = (0.8, 0.3, 0.05)
        # self.table_offset = (-0.2, -0.35, 0.8)
        self.table_offset = (-0.2, 0.5, 0.8)

        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = True

        # whether to use ground-truth object states
        self.use_object_obs = True

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.context_cfg = context_cfg
        self.eef_start_region_cfg = eef_start_region_cfg

        self.fk_sim = None
        self._klampt_world = klampt.WorldModel()
        root = '/home/aries/research/recovery-skills'
        self._klampt_world.loadRobot(
            # to_absolute_path("assets/franka_description/robots/franka_panda.urdf")
            join(root, "assets/franka_description/robots/franka_panda.urdf")
        )
        # self._klampt_world.loadTerrain(to_absolute_path("assets/plane.off"))
        self._klampt_world.loadTerrain(join(root, "assets/plane.off"))
        self._klampt_robot = self._klampt_world.robot(0)
        self._klampt_planning_link = self._klampt_robot.link("panda_eef")
        self._klampt_root_link = self._klampt_robot.link("panda_link0")
        self._planning_joints = list(range(8))
        self._mujoco_root_link = "robot0_link0"

        super().__init__(
            robots="Panda",
            controller_configs=controller_configs,
            mount_types="default",
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=False,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
        )

        self._orig_obs_uncertainty = {}
        # Add corruption
        if obs_uncertainty:
            self._orig_obs_uncertainty = deepcopy(obs_uncertainty)
            self._obs_uncertainty = deepcopy(obs_uncertainty)
            self._update_observable_uncertainties(obs_uncertainty)
        self.disable_obs_corruption = False

        self.update_sim_timestep(timestep)

    # Env definition stuff
    # ----------------------
    @property
    def action_spec(self):
        """Will need to change for PPO/SAC."""
        low, high = [], []
        for robot in self.robots:
            lo, hi = robot.action_limits
            low, high = np.concatenate([low, lo]), np.concatenate([high, hi])
        return low, high

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the door is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between door handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by door handled
              - Note that this component is only relevant if the environment is using the locked door version

        Note that a successfully completed task (door opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            # Add rotating component if we're using a locked door
            if self.use_latch:
                handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                reward += np.clip(
                    0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25
                )

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    # Abstractions
    def obs(self, ground_truth=False, force_update=False):
        """ground_truth=True disables all corruption"""
        # """ground_truth=True sets low corruption"""

        if self.disable_obs_corruption:
            ground_truth = True
            # for observable in self.observation_names:
                # corrupter = create_gaussian_noise_corrupter(mean=0.0,
                                                            # std=0.005)
                # self.modify_observable(observable, 'corrupter', corrupter)

        if force_update and self._orig_obs_uncertainty:
            self.update_state_estimate()

        observations = (
            self.viewer._get_observations(force_update=force_update)
            if self.viewer_get_obs
            else self._get_observations(force_update=force_update,
                                        ground_truth=ground_truth)
        )
        return observations

    def state(self, obs=None, ground_truth=False):
        """Uniquely defines a state. Useful for setting simulation state."""
        if obs is None:
            obs = self.obs(ground_truth=ground_truth)
        state = State(self.state_vars, self.state_var_ndims).from_dict(obs)
        return state

    def rl_state(self, obs=None, ground_truth=False):
        """
        Abstraction more amenable for learning RL policies.
        Does NOT contain context variables.
        """
        if obs is None:
            obs = self.obs(ground_truth=ground_truth)
        rl_state = State(self.rl_state_vars, self.rl_state_var_ndims).from_dict(obs)
        # state = self.create_state().from_dict(obs)
        # state = self.state_to_rl_state(state)
        return rl_state

    def ground_truth_rl_state(self):
        obs = self.obs(ground_truth=True)
        gt_rl_state = State(self.rl_state_vars, self.rl_state_var_ndims).from_dict(obs)
        return gt_rl_state

    def context(self, obs=None):
        if obs is None:
            obs = self.obs()

        context = State(self.context_vars, self.context_var_ndims).from_dict(obs)
        return context

    def rl_state_and_context(self, obs=None):
        if obs is None:
            obs = self.obs()
        return self.rl_state(obs), self.context(obs)

    @property
    def ndim(self):
        return sum(self.state_var_ndims)

    @property
    def nvars(self):
        return len(self.state_vars)

    # State estimation
    def update_state_estimate(self):
        logger.debug("  Updating state estimate")
        for observable, val in self._obs_uncertainty.items():
            val['std'] /= 2
            logger.debug(f"  Observable {observable} corruption std set to {val.std}")
            corrupter = create_gaussian_noise_corrupter(mean=val.mean,
                                                        std=val.std)
            self.modify_observable(observable, 'corrupter', corrupter)

    def _update_observable_uncertainties(self, obs_uncertainty):
        for observable, val in obs_uncertainty.items():
            corrupter = create_gaussian_noise_corrupter(mean=val.mean,
                                                        std=val.std)
            self.modify_observable(observable, 'corrupter', corrupter)


    # Scene construction
    # --------------------
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        # mujoco_arena = TableArena(
            # table_full_size=self.table_full_size,
            # table_offset=self.table_offset,
        # )
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        # initialize objects of interest
        self.door = DoorObject(
            name="Door",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.door)
        else:
            if self.context_cfg is None:
                self.context_cfg = {
                    'door': {'x_range': [0.35 + 0.05, 0.35 + 0.1],
                             'y_range': [0.13 + 0.03, 0.13 + 0.08],
                             'z_range': [-0.05, 0.05],
                             'rotation': {
                                 'axis': 'z',
                                 'range': (-np.pi - 0.25, -np.pi)
                             }
                             }
                }
                # rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0),
                # x_range=[0.32, 0.34],
                # y_range=[1.0 -0.01, 1.0 + 0.01],

                # rotation=(-np.pi - 0.5, -np.pi - 0.25),
                # rotation=(-np.pi, -np.pi),
                # rotation=(-np.pi + -np.pi / 2.0 - 0.25, -np.pi + -np.pi / 2.0),

            door_cfg = self.context_cfg['door']
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.door,
                x_range=door_cfg['x_range'],
                y_range=door_cfg['y_range'],
                z_range=door_cfg['z_range'],
                rotation=door_cfg['rotation']['range'],
                rotation_axis=door_cfg['rotation']['axis'],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.door,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        # Bodies
        self.object_body_ids = dict()
        self.object_body_ids["door_root"] = self.sim.model.body_name2id(
            self.door.root_body
        )
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(
            self.door.frame_body
        )
        self.object_body_ids["latch"] = self.sim.model.body_name2id(
            self.door.latch_body
        )

        # Geoms
        self.door_handle_geom_id = self.sim.model.geom_name2id(self.door.important_sites["handle"])

        # Sites
        self.door_handle_site_id = self.sim.model.site_name2id(
            self.door.important_sites["handle"]
        )

        # Joints
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])

        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(
                self.door.joints[1]
            )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def door_pos(obs_cache):
                # Door root frame
                return np.array(
                    self.sim.data.body_xpos[self.object_body_ids["door_root"]]
                )

            @sensor(modality=modality)
            def door_theta(obs_cache):
                # frame
                quat = T.convert_quat(
                    np.array(
                        self.sim.data.body_xquat[self.object_body_ids["door_root"]]
                    ),
                    "xyzw",
                )
                theta = T.quat2axisangle(quat)[2]  # about z axis
                return theta

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def handle_ori(obs_cache):
                return self._handle_xquat

            @sensor(modality=modality)
            def handle_center_of_rotation_pos(obs_cache):
                return self.sim.data.get_site_xpos(
                    self.door.correct_naming("handle_center_of_rotation")
                )

            @sensor(modality=modality)
            def door_to_eef_pos(obs_cache):
                return (
                    obs_cache["door_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "door_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def handle_to_eef_pos(obs_cache):
                return (
                    obs_cache["handle_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                return self.sim.data.qpos[self.hinge_qpos_addr]

            @sensor(modality=modality)
            def handle_dims(obs_cache):
                return np.array(self.sim.model.geom_size[self.door_handle_geom_id])

            sensors = [
                door_pos,
                door_theta,
                handle_pos,
                # door_to_eef_pos,
                # handle_to_eef_pos,
                hinge_qpos,
                handle_dims,
                handle_center_of_rotation_pos,
            ]
            # names = [s.__name__ for s in sensors]
            names = [
                "door:pose/position",
                "door:pose/theta",
                "handle:pose/position",
                # 'handle_to_eef:pose/position'
                "hinge:pose/theta",
                "handle:dims",
                "handle_center_of_rotation:pose/position"
            ]

            # Also append handle qpos if we're using a locked door version with rotatable handle
            if self.use_latch:

                @sensor(modality=modality)
                def handle_qpos(obs_cache):
                    return np.array([self.sim.data.qpos[self.handle_qpos_addr]])

                sensors.append(handle_qpos)
                names.append("handle:pose/theta")

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            door_pos, door_quat, obj = object_placements[self.door.name]
            # door_quat = np.array([np.cos(np.pi/4), 0, 0 , -np.sin(np.pi/4)])
            # ic(door_quat)
            door_body_id = self.object_body_ids["door_root"]
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat

    def step(self, action):
       obs, reward, done, info = super().step(action)
       obs = self.obs()
       return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.reset_state_estimation()
        obs = self.obs()

        if self.deterministic_reset:
            pass
        else:
            obs = self.reset_eef(obs)

        return obs

    def reset_state_estimation(self):
        if self._orig_obs_uncertainty:
            self._obs_uncertainty = deepcopy(self._orig_obs_uncertainty)
            self._update_observable_uncertainties(self._obs_uncertainty)

    def reset_eef(self, obs):
        def sample_eef_perturbation():
            x = sample(self.eef_start_region_cfg.x)
            y = sample(self.eef_start_region_cfg.y)
            z = sample(self.eef_start_region_cfg.z)

            roll = sample(self.eef_start_region_cfg.roll)
            pitch = sample(self.eef_start_region_cfg.pitch)
            yaw = sample(self.eef_start_region_cfg.yaw)

            eef_pos = [x, y, z]
            eef_quat = T.mat2quat(T.euler2mat([roll, pitch, yaw]))
            return eef_pos, eef_quat

        eef_pos, eef_quat = obs['robot_eef:pose/position'], obs['robot_eef:pose/quat']
        eef_pos_perturb, eef_quat_perturb = sample_eef_perturbation()
        eef_pos = eef_pos + eef_pos_perturb
        eef_quat = T.quat_multiply(eef_quat, eef_quat_perturb)

        eef_pos = eef_pos.tolist()
        eef_axisangle = T.quat2axisangle(eef_quat).tolist()
        action = eef_pos + eef_axisangle + [0.0]
        for _ in range(25):
            obs, _, _, _ = self.step(action)

        return obs

    def reset_from_state(self, state, use_door_pose=True):
        """
        pillar_state: element of the state space
        """
        self.deterministic_reset = True
        self.reset()

        # objects
        if use_door_pose:
            door_pos = np.array(state["door:pose/position"])
            door_angle = state["door:pose/theta"]
            door_quat = np.array([np.cos(door_angle / 2), 0, 0, np.sin(door_angle / 2)])
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat

        else:
            self.reset_handle_pose(state["handle:pose/position"])

        # hinge angle
        self.sim.data.qpos[self.hinge_qpos_addr] = state["hinge:pose/theta"]

        # handle angle
        self.sim.data.qpos[self.handle_qpos_addr] = state["handle:pose/theta"]

        # robot
        if 'sim_state' in state:
            self.sim.set_state(state['sim_state'])

        else:
            robot_joints = np.array(state["robot:arm/joints"])
            arm_joints = robot_joints[:7]
            gripper_joints = robot_joints[7:]
            # Change intial_qpos
            # XXX modifies the default init pose of the arm
            self.robots[0].init_qpos = arm_joints
            self.robots[0].gripper.init_qpos = gripper_joints
            self.robots[0].reset()

        self.deterministic_reset = False

        self.sim.forward()
        self.sim.step()

        # Return new observations
        return self.obs(force_update=True)

    def reset_handle_pose(self, handle_pos, handle_rot=None):

        t_door_world = deepcopy(self.sim.data.get_body_xpos(self.door.root_body))
        R_door_world = deepcopy(self.sim.data.get_body_xmat(self.door.root_body))
        T_door_world = RigidTransform(translation=t_door_world,
                                       rotation=R_door_world,
                                       from_frame='door')
        # center of handle
        t_handle_world = deepcopy(self.sim.data.get_site_xpos('Door_handle'))
        R_handle_world = deepcopy(self.sim.data.get_site_xmat('Door_handle'))
        T_handle_world = RigidTransform(translation=t_handle_world,
                                        rotation=R_handle_world,
                                        from_frame='handle')
        # T_handle_door = T_world_handle.inverse() * T_world_door
        # T_door_handle = T_world_door.inverse() * T_world_handle
        # this stasys fixed
        T_handle_door = T_door_world.inverse() * T_handle_world

        new_t_handle_world = handle_pos
        if handle_rot is None:
            handle_rot = R_handle_world

        new_R_handle_world = handle_rot
        new_T_handle_world = RigidTransform(translation=new_t_handle_world,
                                            rotation=new_R_handle_world,
                                            from_frame='handle')
        # new_T_world_door = new_T_world_handle * T_handle_door
        new_T_door_world = new_T_handle_world * T_handle_door.inverse()

        # set the door transform in the model
        door_body_id = self.sim.model.body_name2id(self.door.root_body)
        self.sim.model.body_pos[door_body_id] = new_T_door_world.translation
        self.sim.model.body_quat[door_body_id] = new_T_door_world.quaternion

        self.sim.forward()
        self.sim.step()

        return self.obs()

    # Actions
    # --------
    def _pre_action(self, action, policy_step=False):
        """
        Convert the high level actions into robot control commands.
        """
        # Verify that the action is the correct dimension
        assert (
            len(action) == self.action_dim
        ), "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

        # Update robot joints based on controller actions
        cutoff = 0
        for idx, robot in enumerate(self.robots):
            robot_action = action[cutoff : cutoff + robot.action_dim]
            robot.control(robot_action, policy_step=policy_step)
            cutoff += robot.action_dim

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        info["is_solved"] = self._check_success()
        info["task_id"] = "FrankaDoorLatched"
        info["timestep"] = self.timestep
        info["gt_obs"] = self.obs(ground_truth=True)
        return reward, done, info

    def _check_success(self, obs=None):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        if obs:
            hinge_qpos = obs['hinge:pose/theta']
        else:
            hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        return hinge_qpos > 0.3
        # return hinge_qpos > np.pi/6

    # Utils
    # -------
    def render(self, **kwargs):
        self.viewer.render()

    def render_by(self, num_steps=100):
        for _ in range(num_steps):
            self.render()

    def set_obs_corruption(self, val):
        self.disable_obs_corruption = (not val)

    def _setup_fk_sim(self):
        if not self.fk_sim:
            # for forward kinematics
            robot_arm_model = load_model_from_xml(self.model.get_xml())
            self.fk_sim = MjSim(robot_arm_model)
            # these are two different locations
            # site is attached to gripper
            self.eef_site_id = self.robots[0].eef_site_id
            # body is before the gripper
            self.eef_name = self.robots[0].robot_model.eef_name

    def fk(self, joints):
        self._setup_fk_sim()

        joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        joint_addrs = [
            self.fk_sim.model.get_joint_qpos_addr(joint) for joint in joint_names
        ]
        self.fk_sim.data.qpos[joint_addrs] = joints[:7]
        self.fk_sim.forward()

        ee_pos = np.array(self.fk_sim.data.site_xpos[self.eef_site_id])
        # we are interested in the arm's orientation, not the gripper's
        ee_quat = self.fk_sim.data.get_body_xquat(self.eef_name)
        ee_quat = T.convert_quat(ee_quat, to="xyzw")

        return ee_pos, ee_quat

    def ik(self, eef_pos, eef_quat):
        root_pos = self.sim.data.get_body_xpos(self._mujoco_root_link)
        root_mat = self.sim.data.get_body_xmat(self._mujoco_root_link)

        # set in klampt
        self._klampt_root_link.setParentTransform(root_mat.flatten(order="F"), root_pos)

        goal_mat = T.quat2mat(eef_quat).flatten(order="F")

        ik_obj = ik.objective(
            self._klampt_planning_link, R=list(goal_mat), t=list(eef_pos)
        )
        solver = ik.solver(ik_obj, iters=10000, tol=0.01)
        solver.setActiveDofs(self._planning_joints)
        solved = solver.solve()

        return solved, self._klampt_robot.getConfig()

    # def global_to_local_actions(self, params):
        # actions = params.reshape(-1, 7)
        # for action in actions:
            # pos = action[:3]
            # axisangle = action[3:6]
            # gripper = action[6]
            # local_action = RigidTransform(translation=pos,
                                            # rotation=T.quat2mat(T.axisangle2quat(axisangle)),
                                            # from_frame='franka_tool',
                                            # to_frame='franka_tool')
            # global_action =  eef_pose * local_action
            # global_actions.append((global_action, gripper))


    def get_warm_initialization(self, start_state, goal_state, method="interpolate"):
        nwaypoints = 3
        if method == "interpolate":
            actions = FrankaDoorEnv.interpolate_ee_poses(
                # self.state_to_rl_state(start_state), goal_state, nwaypoints
                # FIXME
                # self.state_to_rl_state(start_state), self.state_to_rl_state(goal_state),
                self.state_to_rl_state(start_state), goal_state,
                nwaypoints
            )
            gripper_actions = np.zeros(len(actions))
            mean_init = np.column_stack([actions, gripper_actions])
            mean_init = mean_init.flatten()

        elif method == "motion_plan":
            plan = self.get_motion_plan(start_state, goal_state, nwaypoints,
                                        cartesian=True)
            gripper_actions = np.zeros(len(plan))
            mean_init = np.column_stack([plan, gripper_actions])
            mean_init = np.concatenate(mean_init)

        else:
            raise NotImplementedError

        # TODO
        # gripper_pos = start_state['']

        var_init = np.diag(
            np.tile(
                # [0.01 ** 2] * 3 + [(np.pi/32) ** 2] * 3 + [1],
                [0.02 ** 2] * 3 + [(np.pi/32) ** 2] * 3 + [0.5],
                # [0.005 ** 2] * 3 + [(np.pi/132) ** 2] * 3 + [0.5],
                # [0.001 ** 2] * 3 + [(np.pi/126) ** 2] * 3 + [0],
                nwaypoints,
            )
        )

        # mean_init = global_to_local_actions(mean_init)

        if var_init.shape != (mean_init.size, mean_init.size):
            __import__('ipdb').set_trace()

        return mean_init, var_init

    def get_motion_plan(self, start_state, goal_state, nwaypoints,
                        cartesian=True, local_frame=True):
        """
        Arguments:
            start_state: robot joint config
            goal_state: end-effector pose if cartesian else joint config
        """
        world, robot = self._klampt_world, self._klampt_robot

        # set link0 pose
        klampt_planning_link = self._klampt_planning_link
        klampt_root_link = self._klampt_root_link
        mujoco_root_link = self._mujoco_root_link

        # get from mujoco
        root_pos = self.sim.data.get_body_xpos(mujoco_root_link)
        root_mat = self.sim.data.get_body_xmat(mujoco_root_link)

        # set in klampt
        klampt_root_link.setParentTransform(root_mat.flatten(order="F"), root_pos)

        # plan in klampt
        planning_joints = self._planning_joints
        start_joints = start_state["robot:arm/joints"][:7]

        base_joints = [0.0]
        gripper_joints = [0.0, 0.0, 0.01, 0.01]

        qstart = base_joints + list(start_joints) + gripper_joints
        robot.setConfig(qstart)

        goal_dict = goal_state

        if cartesian:
            goal_pos, goal_quat = (
                goal_dict["robot_eef:pose/position"],
                goal_dict["robot_eef:pose/quat"],
            )

            solved, config = self.ik(goal_pos, goal_quat)

            if solved:
                ic("Solved")
                qgoal = robot.getConfig()
                vis.add("robot_ik", robot)
                vis.setColor(("robot_ik", "panda_hand"), 0, 0, 1)
                mat, pos = klampt_planning_link.getTransform()
                quat = T.mat2quat(np.array(mat).reshape((3, 3), order="F"))
                ic("Solution pose")
                ic(quat)
                ic(pos)

            else:
                logger.warning("Could not find valid IK solution")
                raise RuntimeError
        else:
            goal_joints = goal_state.as_ordered_dict()["robot:arm/joints"][:7]
            qgoal = base_joints + list(goal_joints) + gripper_joints

        # Plan to found config
        robot.setConfig(qstart)
        planner = robotplanning.planToConfig(
            world, robot, qgoal, movingSubset=planning_joints
        )
        if not planner:
            raise RuntimeError

        # planner = robotplanning.planToCartesianObjective(world, robot,
        # iktarget,
        # movingSubset=planning_joints)

        for i in range(10):
            planner.planMore(10000)
            plan = planner.getPath()
            ic("Plan length: ", len(plan))
            if plan:
                break

        def vis_plan(plan, robot):
            ic(plan)
            vis.show()
            while vis.shown():
                for config in plan:
                    robot.setConfig(config)
                    time.sleep(1)

        plan = [planner.space.project(config) for config in plan]
        plan = [planner.space.lift(planner.space.interpolate(plan[0], plan[1], u))
                for u in np.linspace(0, 1, nwaypoints + 1)]
        # vis_plan(plan, robot)

        if cartesian:
            eef_plan = []
            for config in plan:
                robot.setConfig(config)
                eef_mat, eef_pos = klampt_planning_link.getTransform()
                eef_axisangle = T.quat2axisangle(T.mat2quat(
                    np.reshape(eef_mat, (3, 3), order='F')))
                eef_plan.append(np.concatenate([eef_pos, eef_axisangle]))
            plan = eef_plan

            if local_frame:
                global_poses = []
                for state in plan:
                    pose = RigidTransform(translation=state[:3],
                                          rotation=T.quat2mat(T.axisangle2quat(state[3:])))
                    global_poses.append(pose)

                local_poses = []
                start_ee_pose = global_poses[0]
                for global_pose in global_poses:
                    local_pose = start_ee_pose.inverse() * global_pose
                    local_poses.append(local_pose)

                eef_plan = []
                for pose in local_poses:
                    eef_plan.append(np.concatenate([pose.translation,
                                                    pose.axis_angle]))

                plan = eef_plan

        else:
            plan = [planner.space.project(config) for config in plan]

        return plan[1:] # skip the start state

    def state_to_rl_state(self, state):
        """Converts a sim state to an RL state by calling FK"""
        self.reset_from_state(state)
        rl_state = self.rl_state()
        # joints = state.as_ordered_dict()["robot:arm/joints"]
        # ee_pose, ee_quat = self.fk(joints)
        # state_dict = state.as_ordered_dict()
        # state_dict["robot_eef:pose/position"] = ee_pose
        # state_dict["robot_eef:pose/quat"] = ee_quat
        # state_dict["robot_eef:gripper/position"] = joints[7]

        # rl_state = State(self.rl_state_vars, self.rl_state_var_ndims).from_dict(
            # state_dict
        # )

        return rl_state

    def rl_state_to_state(self, rl_state):
        ee_pos = rl_state['robot_eef:pose/position']
        ee_quat = rl_state['robot_eef:pose/quat']
        joints = self.ik(ee_pos, ee_quat)
        dict = rl_state.as_ordered_dict()
        dict['robot:arm/joints'] = joints
        state = self.create_state().from_dict(dict)
        return state

    def _handle_is_grasped(self):
        # FIXME
        if self._gripper_to_handle() < 0.05:
            return True
        else:
            return False

    @property
    def _handle_xpos(self):
        """
        Grabs the position of the door handle handle.

        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _handle_xquat(self):
        """
        Grabs the orientation of the door handle handle.

        Returns:
            np.array: Door handle orientation
        """
        return T.mat2quat(
            np.reshape(
                self.sim.data.site_xmat[self.door_handle_site_id],
                (3, 3)
            )
        )

    @property
    def _gripper_to_handle(self):
        """
        Calculates distance from the gripper to the door handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._handle_xpos - self._eef_xpos

    # Viz stuff
    # -----------
    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.door.important_sites["handle"],
                target_type="site",
            )

    @staticmethod
    def state_to_context(state):
        context = State(
            FrankaDoorEnv.context_vars, FrankaDoorEnv.context_var_ndims
        ).from_dict(state.as_ordered_dict())
        return context

    # Static methods
    @staticmethod
    def interpolate_ee_poses(start_state, goal_state, nwaypoints,
                             local_frame=True):
        # make sure the contexts are identical
        # start_context = FrankaDoorEnv.state_to_context(start_state)
        # goal_context = FrankaDoorEnv.state_to_context(goal_state)
        # assert start_context.is_close(goal_context), "Start and goal have different contexts!"

        # convert from world state to robot actions
        # for FK
        # ee_quat = T.convert_quat(ee_quat, to='wxyz')
        start_dict, goal_dict = (
            start_state.as_ordered_dict(),
            goal_state.as_ordered_dict(),
        )
        # ee_pos, ee_quat = self.fk(start_dict['robot:arm/joints'])

        ee_pos, ee_quat = (
            start_dict["robot_eef:pose/position"],
            start_dict["robot_eef:pose/quat"],
        )

        start_ee_pose = RigidTransform(
            translation=ee_pos,
            rotation=RigidTransform.rotation_from_quaternion(T.convert_quat(ee_quat,
                                                                            'wxyz')),
        )

        # ee_pos, ee_quat = FrankaDoorEnv.fk(goal_state)
        # ee_quat = T.convert_quat(ee_quat, to='wxyz')
        ee_pos, ee_quat = (
            goal_dict["robot_eef:pose/position"],
            goal_dict["robot_eef:pose/quat"],
        )
        goal_ee_pose = RigidTransform(
            translation=ee_pos,
            rotation=RigidTransform.rotation_from_quaternion(T.convert_quat(ee_quat,
                                                                            'wxyz')),
        )

        interp_ee_poses = RigidTransform.linear_trajectory_to(
            start_ee_pose, goal_ee_pose, traj_len=nwaypoints + 1
        )

        local_interp_ee_poses = []

        if local_frame:
            for global_pose in interp_ee_poses:
                local_pose = start_ee_pose.inverse() * global_pose
                local_interp_ee_poses.append(local_pose)
            interp_ee_poses = local_interp_ee_poses

        interp_ee_poses = interp_ee_poses[1:]
        actions = []
        for pose in interp_ee_poses:
            action = np.concatenate([pose.translation, pose.axis_angle])
            actions.append(action)
        actions = np.array(actions)
        return actions

    @staticmethod
    def create_state():
        return State(FrankaDoorEnv.state_vars,
                     FrankaDoorEnv.state_var_ndims)

    @staticmethod
    def create_rl_state():
        return State(FrankaDoorEnv.rl_state_vars,
                     FrankaDoorEnv.rl_state_var_ndims)
