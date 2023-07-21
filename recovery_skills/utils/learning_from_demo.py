from icecream import ic
import numpy as np
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils.input_utils import input2action
from recovery_skills.skills import IdentityEESpaceSkill
from recovery_skills.graph.goal_constraint import GoalConstraint


def record_demo(env, cfg, controller_cfg, seed):
    # env_cfg.render_camera  = "agentview"
    env_cfg = cfg.env
    controller_cfg["control_delta"] = True
    env = VisualizationWrapper(env, indicator_configs=None)
    if cfg.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=cfg.keyboard.pos_sensitivity,
            rot_sensitivity=cfg.keyboard.rot_sensitivity,
        )
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)

        done = False

        # while not done:
        obs = env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        (device_traj, action_traj, obs_traj, state_traj, context_traj, modes) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        while not done:
            obs = env.obs()
            state = env.state()
            context = []
            obs_traj.append(obs)
            state_traj.append(state)
            context_traj.append(context)

            active_robot = env.robots[0]

            # Get the newest action
            action, grasp, mode = input2action(
                device=device,
                robot=active_robot,
                active_arm="left",
                env_configuration="single-arm-opposed",
            )
            action_traj.append(action)
            modes.append(mode)

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if last_grasp < 0 < grasp:
                if cfg.toggle_camera_on_grasp:
                    cam_id = (cam_id + 1) % num_cam
                    env.viewer.set_camera(camera_id=cam_id)
            # Update last grasp
            last_grasp = grasp

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                if env_cfg.arm == "right":
                    action = np.concatenate([action, rem_action])
                elif env_cfg.arm == "left":
                    action = np.concatenate([rem_action, action])
                else:
                    # Only right and left arms supported
                    ic(
                        "Error: Unsupported arm specified -- "
                        "must be either 'right' or 'left'! Got: {}".format(env_cfg.arm)
                    )
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: env.action_dim]

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            env.render()
    return dict(
        obs=obs_traj,
        states=state_traj,
        contexts=context_traj,
        actions=action_traj,
        modes=modes,
    )


def segment_demo(demo_results):
    actions = demo_results['actions']
    actions = np.stack([action for action in actions if action is not None],
                       axis=0)
    N = actions.shape[0]
    modes = demo_results['modes']
    states = demo_results['states']

    subskills = []
    n = 0
    for i in range(1, N):
        if modes[i] != modes[i-1]:
            goal = states[i]
            policy = actions[n:i, :]
            skill = IdentityEESpaceSkill()
            skill.update_policy(policy)
            skill.goal_constraint = GoalConstraint(goal)
            skill.start_state = states[n]
            subskills.append(skill)
            n = i

    # terminal skill
    goal = states[N-1]
    policy = actions[n:N, :]
    skill = IdentityEESpaceSkill()
    skill.update_policy(policy)
    skill.goal_constraint = GoalConstraint(goal)
    skill.start_state = states[n]
    subskills.append(skill)
    return subskills

