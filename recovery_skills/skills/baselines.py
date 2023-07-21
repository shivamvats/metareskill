from .robot_skill import RobotSkill
from recovery_skills.envs.franka_door_env import FrankaDoorEnv
from recovery_skills.utils.transforms import *


class RetrySkill(RobotSkill):
    """Retry the failed skill"""
    def __init__(self, skill):
        super().__init__()
        self.retry_skill = skill

    def precondition_satisfied(self, state, context=None):
        return True

    def apply(self, env, obs, context=None, render=False, interpolate=False):
        return self.retry_skill.apply(env, obs, context, render, interpolate)


class GoToPrevStateSkill(RobotSkill):
    """Go to the precondition of the failed skill."""
    def __init__(self, prev_skill, eval_env):
        super().__init__()
        self.prev_skill = prev_skill
        self.eval_env = eval_env

    def precondition_satisfied(self, state, context=None):
        return True

    def apply(self, env, obs, context=None, render=False, interpolate=False):
        time_per_action = self.time_per_action

        TIMESTEPS_PER_ACTION = int(time_per_action / env.control_timestep)

        target_precond = self.prev_skill.preconds
        target_state = self.eval_env.state_to_rl_state(
            target_precond.sample(cls=FrankaDoorEnv, n=1)[0])
        target_ee_pos = target_state['robot_eef:pose/position']
        target_quat = target_state['robot_eef:pose/quat']
        target_ori = T.quat2axisangle(target_quat)
        target_grasp = target_state['robot_eef:gripper/position']
        current_grasp = obs['robot_eef:gripper/position']
        actions = []

        if np.sign(current_grasp) == np.sign(target_grasp):
            action = np.concatenate([target_ee_pos, target_ori, [target_grasp]])
            actions.append(action)

        else:
            if np.sign(current_grasp) == -1.0:
                # open
                action1 = np.concatenate([target_ee_pos, target_ori, [current_grasp]])
                action2 = np.concatenate([target_ee_pos, target_ori, [target_grasp]])
                actions = [action1, action2]
            elif np.sign(current_grasp) == 1.0:
                # closed
                action1 = np.concatenate([target_ee_pos, target_ori, [target_grasp]])
                action2 = np.concatenate([target_ee_pos, target_ori, [target_grasp]])
                actions = [action1, action2]

        for action in actions:
            for _ in range(TIMESTEPS_PER_ACTION):
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()

        return obs, rew, done, info


class GoToStartSkill(RobotSkill):
    """Go all the way back to the start."""
    def __init__(self, home_pos, home_quat):
        super().__init__()
        self.home_pos = home_pos
        self.home_ori = T.quat2axisangle(home_quat)

    def precondition_satisfied(self, state, context=None):
        return True

    def apply(self, env, obs, context=None, render=False, interpolate=False):
        time_per_action = self.time_per_action

        TIMESTEPS_PER_ACTION = int(time_per_action / env.control_timestep)

        target_pos = self.home_pos
        target_ori = self.home_ori
        action = np.concatenate([target_pos, target_ori, [1.0]])

        actions = [action]
        for action in actions:
            for _ in range(TIMESTEPS_PER_ACTION):
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()

        return obs, rew, done, info
