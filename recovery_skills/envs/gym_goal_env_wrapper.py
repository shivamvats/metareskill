import logging

from gym.core import GoalEnv
from gym import spaces
import numpy as np
from robosuite.wrappers import GymWrapper


class GymGoalEnvWrapper(GymWrapper):

    """
    This is a wrapper over the `FrankaDoorEnv` env to be used to train
    goal-directed skills. It follows gym's `GoalEnv` API.
    """

    def __init__(self, env, keys=None):
        super().__init__(env, keys)

        obs = self._flatten_obs(self.obs())
        # TODO replace components with state-space + 
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=self.env.goal.shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=self.env.goal.shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs.shape, dtype="float32"
                ),
            )
        )

    # GoalEnv method
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError

    def reset(self):
        ob_dict = self.env.reset()
        return self._obs_to_goal_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        goal_obs = self._obs_to_goal_obs(ob_dict)
        return goal_obs, reward, done, info

    def _obs_to_goal_obs(self, obs_dict):
        """Convert obs dict to GoalEnv compatible flattened obs."""
        achieved_goal = self.env.state(obs_dict)
        goal_obs = {
            "observation": self._flatten_obs(obs_dict),
            "desired_goal": self.env.goal,
            "achieved_goal": self._flatten_state(achieved_goal),
        }
        return goal_obs

    def _flatten_state(self, state):
        flat_state = []
        for key in self.env.state_vars:
            val = np.array(state[key])
            if val.ndim == 0:
                val = [val]
            flat_state.append(val)
        flat_state = np.concatenate(flat_state)
        return flat_state

    # Viz
    # -----
    def visualize(self, vis_settings):
        self.env.visualize(vis_settings=vis_settings)

        # visualize goal
        # TODO
        # target_pos = self.sim.data.get_body_xpos(target.root_body)
