from abc import ABC, abstractmethod

import numpy as np
from recovery_skills.graph.abstraction import angle_weighted_euclidean_distance


class GoalConstraint(ABC):
    """
    Represents an abstract goal constraint to be used in skill chaining and
    policy learning using RL.
    """

    @abstractmethod
    def is_satisfied(self, state):
        pass

    @abstractmethod
    def distance(self, state):
        pass

    def reward(self, state):
        reward = 0.0
        if self.is_satisfied(state):
            reward += 10.0

        dist_to_goal = self.distance(state)
        reward -= 10 * dist_to_goal

        return reward


class StateGoalConstraint(GoalConstraint):

    """
    Represents a goal constraint defined with respect to a state.
    """

    def __init__(
        self,
        goal_state,
        thresh=0.1,
        distance_fn=angle_weighted_euclidean_distance,
        relevant_cols=None,
        is_satisfied_fn=None,
    ):
        """
        relevant_cols: Indices of relecant columns
        """
        self.goal = goal_state
        self.thresh = thresh

        self._goal_arr = self.goal.as_array()
        self._distance_fn = distance_fn

        if not relevant_cols:
            relevant_cols = np.arange(goal_state.ndim)
        # Abstraction
        self.relevant_cols = relevant_cols

        self._is_satisfied_fn = is_satisfied_fn

    def is_satisfied(self, state):
        if self._is_satisfied_fn:
            return self._is_satisfied_fn(self, state)
        else:
            return self.distance(state) < self.thresh

    def distance(self, state):
        return self._distance_fn(state, self.goal)

    def get_closest_goal(self, state=None):
        return self.goal
