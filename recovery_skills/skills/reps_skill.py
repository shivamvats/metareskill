import copy
import logging
import numpy as np

from autolab_core import RigidTransform
from icecream import ic
from autolab_core import RigidTransform as RT
from robosuite.wrappers import VisualizationWrapper, GymWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from .robot_skill import RobotSkill
from ..envs import FrankaDoorSubtaskEnv, FrankaDoorEnv
from ..envs.improved_gym_wrapper import ImprovedGymWrapper
from ..envs import RayActorWrapper, RayActorSubtaskWrapper, RayVecEnvWrapper
from ..graph.goal_constraint import GoalConstraint
from recovery_skills.utils.robosuite import make_env
from recovery_skills.utils import WandbCallback
from recovery_skills.utils.transforms import *
from recovery_skills.utils import solve_env_using_reps, load_reps_params
from recovery_skills.graph.abstraction import *
from recovery_skills.graph.preconditions import PreconditionClassifier

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")


class REPSSkill(RobotSkill):
    """
    The policy is represented as a REPS agent.

    This trains a skill under the assumption of perfect state estimation.
    """

    def __init__(
        self, goal_constraint, env_cfg, policy_type="identity", indicator_cfg="default"
    ):
        super().__init__(goal_constraint)
        self._env_cfg = copy.deepcopy(env_cfg)
        self._env_cfg['obs_uncertainty']['handle:pose/position']['std'] = 0.0
        # logger.warning("Setting handle pose uncertainty to 0")

        self.policy_type = policy_type

        # True is this skill connects directly to the task goal
        self.terminal = False

        if policy_type == "identity":
            self.policy = self._identity_policy
            self.num_steps = 1
            # trained on only one task
            self.start = None

        elif policy_type == "linear":
            # TODO
            self._A, self._b = None, None
            self.policy = self._linear_policy

        self.indicator_cfg = indicator_cfg

    def precondition_satisfied(self, state, context=None):
        if self.preconds:
            return self.preconds.is_satisfied(state, context)
        else:
            return True

    def termcondition_satisfied(self, state, context=None):
        return False

    def make_policy(self, state, context=None):
        raise NotImplementedError

    def apply(
        self,
        env,
        obs,
        context=None,
        render=False,
        interpolate=False,
        deterministic=False,
        local_frame=False,
    ):
        if self.policy_type == "identity":
            if hasattr(env, 'num_envs'):
                size = env.num_envs
                is_vec_env = True
            else:
                size = 1
                is_vec_env = False

            actions, timesteps, sample_info = self.sample_policy(
                size=size, is_vec_env=is_vec_env, deterministic=deterministic)

            # local to global
            def local_to_global(obs, local_actions):
                eef_pose = RigidTransform(translation=obs['robot_eef:pose/position'],
                                        rotation=T.quat2mat(obs['robot_eef:pose/quat']),
                                        from_frame='franka_tool',
                                        to_frame='world')
                global_actions = []
                for action in local_actions:
                    pos = action[:3]
                    axisangle = action[3:6]
                    # axisangle = np.zeros(3)
                    gripper = action[6]

                    local_action = RigidTransform(translation=pos,
                                                rotation=T.quat2mat(T.axisangle2quat(axisangle)),
                                                from_frame='franka_tool',
                                                to_frame='franka_tool')
                    global_action =  eef_pose * local_action
                    global_actions.append((global_action, gripper))

                global_params = []
                for action, gripper in global_actions:
                    params = np.concatenate([action.translation, action.axis_angle,
                                            [gripper]])
                    global_params.append(params)
                return global_params

            if local_frame:
                if is_vec_env:
                    global_actions = []
                    rolled_actions = np.moveaxis(np.array(actions), 1, 0)
                    for env_id, action in enumerate(rolled_actions):
                        global_action = local_to_global(obs[env_id], action)
                        global_actions.append(global_action)
                    global_actions = np.array(global_actions)
                    global_actions = np.moveaxis(global_actions, 1, 0)
                else:
                    global_actions = local_to_global(obs, actions)

                actions = global_actions

            rews = None
            for action, timestep in zip(actions, timesteps):
                for idx, _ in enumerate(range(timestep)):
                    # parallelized
                    obs, rew, done, info = env.step(action)
                    # print(rew)
                    if render:
                        env.render()
                logger.debug(f" Reward: {rew}")
                # logger.debug(f"  Distance: {self.goal_constraint.distance(env.state())}")
                if rews is None:
                    rews = [rew]
                else:
                    rews.append(rew)
        else:
            raise NotImplementedError

        if isinstance(info, dict):
            info['policy_params'] = sample_info['policy_params']
        else:
            # vec env
            for info_env, params in zip(info, sample_info['policy_params']):
                info_env['policy_params'] = params

        rews = np.array(rews)
        # rew = np.sum(rews, axis=0)
        # rew = rews[-1]
        rew = np.sum(rews, axis=0) + 5*rews[-1]
        logger.debug(f" Total Reward: {rew}")

        if hasattr(env, 'obs'):
            obs = env.obs(force_update=True)
        else:
            # ray
            pass
        return obs, rew, done, info

    def train_policy(
        self,
        start,
        controller_cfg,
        train_cfg,
        reps_cfg,
        path_to_pretrained_model=None,
        render=False,
    ):
        self.start = start

        _env = FrankaDoorEnv(controller_configs=controller_cfg,
                             has_renderer=False,
                             )
        _env.set_obs_corruption(False)
        _env.reset_from_state(start)

        if train_cfg.use_ray:
            train_envs = [
                RayActorSubtaskWrapper.remote(
                    goal_constraint=self.goal,
                    # self._env_cfg.obs_vars,
                    controller_configs=controller_cfg,
                    has_renderer=render,
                    horizon=self._env_cfg.horizon,
                    context_cfg=self._env_cfg.context,
                    obs_uncertainty=self._env_cfg.obs_uncertainty,
                    timestep=self._env_cfg.timestep,
                    eef_start_region_cfg=self._env_cfg.eef_start_region,
                )
                for _ in range(train_cfg.num_cpus)
            ]
            train_env = RayVecEnvWrapper(train_envs)

        else:
            train_env = FrankaDoorSubtaskEnv(
                goal_constraint=self.goal,
                controller_configs=controller_cfg,
                has_renderer=render,
                horizon=self._env_cfg.horizon,
                context_cfg=self._env_cfg.context,
                obs_uncertainty=self._env_cfg.obs_uncertainty,
                timestep=self._env_cfg.timestep,
                eef_start_region_cfg=self._env_cfg.eef_start_region,
            )
            # train_env = ImprovedGymWrapper(train_env, keys=self._env_cfg.obs_vars)
        # if render:
        # train_env = VisualizationWrapper(train_env, self.indicator_cfg)

        if path_to_pretrained_model:
            raise NotImplementedError
        else:
            obs = train_env.reset_from_state(start)
            if hasattr(self.goal, 'goal_cache'):
                goal = self.goal.goal_cache
            else:
                goal = self.goal.goal(start, _env)
            rl_goal = _env.state_to_rl_state(goal)

            try:
                (
                    policy_params_mean_init,
                    policy_params_var_init,
                ) = train_env.get_warm_initialization(
                    start,
                    # self.goal.goal(start, _env),
                    # FIXME TEMP hack
                    rl_goal,
                    method="motion_plan",
                )
            except RuntimeError:
                logger.warning("    Motion planner failed")
                (
                    policy_params_mean_init,
                    policy_params_var_init,
                ) = train_env.get_warm_initialization(
                    start,
                    # self.goal.goal(start),
                    rl_goal,
                )

            ic("Initialization: ", policy_params_mean_init)
            convergence_criteria = {
                "env_solved": reps_cfg.min_success,
                "min_num_total_updates": reps_cfg.min_updates
            }
            reps_hyperparams = {
                "rel_entropy_bound": reps_cfg.rel_entropy_bound,
                "min_temperature": reps_cfg.min_temperature,
            }

            reps_converged = False
            solve_info = {}

            # for attempt_id in range(2):
                # try:
            (
                reps_converged,
                policy_params_mean,
                policy_params_var,
                solve_info,
            ) = solve_env_using_reps(
                train_env,
                self,
                policy_params_mean_init,
                policy_params_var_init,
                reps_cfg.rollouts_per_update,
                reps_cfg.max_updates,
                convergence_criteria,
                start_state=start,
                reps_hyperparams=reps_hyperparams,
                verbose= (logger.level == logging.DEBUG),
                render=render,
            )
                    # break
                # except ValueError:
                    # continue

        return reps_converged, solve_info

    def update_policy(self, policy_params, context=None):
        if self.policy_type == "identity":
            self._policy_params_mean = policy_params["mean"]
            self._policy_params_cov = policy_params["cov"]
        else:
            raise NotImplementedError

    def sample_policy(self, size=1, is_vec_env=False, deterministic=True):
        if deterministic:

            policy_params = self._policy_params_mean
            actions, timesteps = self._params_to_actions_timesteps(policy_params)

            if is_vec_env:
                policy_params = np.tile(policy_params, (size, 1, 1))
                actions = np.tile(actions, (size, 1, 1))

        else:
            policy_params = np.random.multivariate_normal(
                mean=self._policy_params_mean, cov=self._policy_params_cov, size=size
            )
            if not is_vec_env and size == 1:
                policy_params = policy_params[0]
                actions, timesteps = self._params_to_actions_timesteps(policy_params)
            else:
                actions = []
                for params in policy_params:
                    _actions, _timesteps = self._params_to_actions_timesteps(params)
                    actions.append(_actions)
                timesteps = _timesteps

        if is_vec_env:
            # num_envs x num_actions x num_params
            actions = np.array(actions)
            # num_actions x num_envs x num_params
            actions = np.swapaxes(actions, 0, 1)

        info = {'policy_params': policy_params}

        return actions, timesteps, info

    def _params_to_actions_timesteps(self, policy_params):
        xyz_axisangle_actions = policy_params.reshape(-1, 7)

        # separate eef and gripper motions
        actions = []
        for action in xyz_axisangle_actions:
            xyz_axisangle_action = copy.deepcopy(action)
            # gripper_action = np.zeros(7)
            xyz_axisangle_action[-1] = 0
            gripper_action = copy.deepcopy(action)

            actions.append(xyz_axisangle_action)
            actions.append(gripper_action)

        actions = np.array(actions)
        nactions = len(actions)
        # timesteps = [200 // nactions] * nactions
        timesteps = [25, 20] * (nactions // 2)
        return actions, timesteps

    # @property
    def params(self):
        if self.policy_type == "identity":
            return self._policy_params_mean, self._policy_params_cov

    def train_precondition(self, states, y):
        self.preconds = PreconditionClassifier(states, y)

    def train_termcondition(self, **kwargs):
        raise NotImplementedError

    def get_warm_initialization(self, start_state):
        """
        Initialize with linear interpolation to the goal state.
        """
        goal_state = self.goal.get_closest_goal(start_state)

    def _identity_policy(self, x):
        return self._actions

    def _linear_policy(self, x):
        return self._A * x + self._b
