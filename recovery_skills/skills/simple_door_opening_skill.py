from copy import deepcopy
from icecream import ic
import logging
import numpy as np

from robosuite.utils.transform_utils import *
from .robot_skill import RobotSkill
# from .identity_policy import IdentityEESpacePolicy

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class SimpleDoorOpeningSkill(RobotSkill):

    """A simple door opening skill with hard-coded actions in the ee space."""

    def __init__(self, nactions=3, actions=None):
        super().__init__()
        if actions is None:
            self._actions = np.array(
                [
                    [0, -0.2, 0.1, 0, 0, 0, 0],
                    [0, -0.2, 0, 0, 0, 0, 0],
                    [0, -0.2, -0.2, 0, 0, 0, 0],
                    # [0, 0.5, 0, 0, 0, 0, 0],
                ]
            )
        else:
            self._actions = actions
        self._timesteps_per_action = 100
        self._timesteps = np.linspace(
            self._timesteps_per_action, self._timesteps_per_action * nactions, nactions
        )

    def precondition_satisfied(self, state, context):
        return True

    def termcondition_satisfied(self, state, context):
        curr_timestep = context["timestep"]
        if curr_timestep - self._start_timestep > self._timesteps[-1]:
            return True
        else:
            return False

    def apply(self, env, obs, context):
        obs = env.unflatten_obs(obs)

        self._start_timestep = 0
        context = {'timestep': 0}
        t = 0
        handle_pos = obs['handle_pos']

        x, y, z = 0, 1, 2
        logger.info(f"Action: {self._actions[0, :3]}")
        robot_pos = obs['robot0_eef_pos']
        robot_orien = quat2axisangle(obs['robot0_eef_quat'])
        target_pos = deepcopy(robot_pos)
        target_orien = np.array([1, 0, 0])*np.pi/2
        action = np.concatenate([target_pos, target_orien, [-1]])
        for _ in range(self._timesteps_per_action):
            obs, reward, done, info = env.step(action)
            env.render()
            t += 1

        # logger.info(f"Action: {self._actions[2,:3]}")
        obs = env.unflatten_obs(obs)
        robot_pos = obs['robot0_eef_pos']
        robot_orien = quat2axisangle(obs['robot0_eef_quat'])
        target_pos = deepcopy(handle_pos)
        target_pos[z] -= 0.04
        target_orien = np.array([1, 0, 0])*np.pi/2
        action = np.concatenate([target_pos, target_orien, [-1]])
        for _ in range(self._timesteps_per_action):
            obs, reward, done, info = env.step(action)
            env.render()
            t += 1

        # logger.info(f"Action: {self._actions[2, :3]}")
        obs = env.unflatten_obs(obs)
        robot_pos = obs['robot0_eef_pos']
        robot_orien = quat2axisangle(obs['robot0_eef_quat'])
        target_pos = deepcopy(robot_pos)
        target_orien = np.array([1, 0, 0])*np.pi/2
        action = np.concatenate([target_pos, target_orien, [1]])
        for _ in range(self._timesteps_per_action):
            obs, reward, done, info = env.step(action)
            env.render()
            t += 1
            # logger.info(f"  Timestep: {t}")

        # logger.info(f"Action: {self._actions[3, :3]}")
        obs = env.unflatten_obs(obs)
        robot_pos = obs['robot0_eef_pos']
        robot_orien = quat2axisangle(obs['robot0_eef_quat'])
        target_pos = deepcopy(robot_pos)
        target_pos[x] += 0.05
        target_pos[z] -= 0.1
        # target_pos[y] += 0.02
        target_orien = np.array([1, 0, 0])*np.pi/2
        action = np.concatenate([target_pos, target_orien, [1]])
        for _ in range(self._timesteps_per_action):
            obs, reward, done, info = env.step(action)
            env.render()
            t += 1

        logger.info("  Skill done")

        obs = env.unflatten_obs(obs)

        return obs, reward, done, info

    def make_policy(self, state, context=None):
        pass

    def update_policy(self, new_params, context=None):
        self._actions = np.array(new_params).reshape(-1, 7)

    @property
    def params(self):
        return self._actions.flatten()

    @property
    def num_steps(self):
        return self._timesteps[-1]
