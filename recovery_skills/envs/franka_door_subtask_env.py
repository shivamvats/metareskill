import copy
import logging
from icecream import ic
import numpy as np
from .franka_door_env import FrankaDoorEnv

logger = logging.getLogger(__name__)


class FrankaDoorSubtaskEnv(FrankaDoorEnv):

    """
    This is a thin wrapper over the `FrankaDoor` env to be used to train
    goal-directed skills.

    The reward function is defined using the `GoalConstraint`.
    """

    def __init__(
        self,
        goal_constraint=None,
        terminal=False,
        **kwargs
    ):
        self.goal_constraint = goal_constraint
        super().__init__(**kwargs)

    def reward(self, action=None):
        """Compute reward using the given goal constraint."""

        if self.goal_constraint is not None:
            gt_obs = self.obs(ground_truth=True)
            reward = self.goal_constraint.reward(gt_obs)
            # state = self.state()
            # reward = self.goal_constraint.reward(state)

            # Scale reward if requested
            if self.reward_scale is not None:
                reward *= self.reward_scale / 1.0

        else:
            reward = super().reward(action)

        return reward

    def step(self, action):
        """
        Repeat the same action for better stability
        """

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def distance(self, state):
        return self.goal_constraint.distance(state)

    def _check_success(self):
        """
        Check if goal is satisfied

        Returns:
            bool: True if goal is satisfied
        """

        if self.goal_constraint is not None:
            # success = (self.goal_constraint.is_satisfied(self.rl_state())
            success = (self.goal_constraint.is_satisfied(self.state())
                    or self.check_task_success())
        else:
            success = super()._check_success()

        return success

    def check_task_success(self, obs=None):
        """Check if door opened or not."""
        return super()._check_success(obs)

    def set_goal_constraint(self, goal_constraint):
        self.goal_constraint = goal_constraint

    # @property
    # def abstraction(self):
        # """
          # An abstract state space consisting only the variables relevant to
          # thet task.
        # """
        # return self._goal_constraint.abstraction

    # def abstract_state(self, obs=None):
        # state = self.state(obs)
        # flattened_state = flatten_state_dict(state)
        # abs_state = State(self.abstraction).from_dict(flattened_state)
        # return abs_state

