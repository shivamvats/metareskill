from abc import ABC, abstractmethod

import numpy as np


class RobotSkill(ABC):

    """
    Base class for all robot skills.

    Accepts externally defined functions to overload the default methods.
    """

    def __init__(self, goal_constraint=None):
        self.goal_constraint = goal_constraint
        self.goal_id = None
        self.num_steps = 0
        self.preconds = None

        self.time_per_action = 1.25

        # Implemented externally
        self._precondition_satisfied_fn = None
        self._termcondition_satisfied_fn = None
        self._apply_fn = None

    def precondition_satisfied(self, state, context=None):
        if self._precondition_satisfied_fn:
            return self._precondition_satisfied_fn(self, state, context)
        else:
            raise NotImplementedError

    def precondition_satisfied_vec(self, states, contexts=None):
        if contexts is None:
            contexts = [None] * len(states)
        return np.array(
            (
                self.precondition_satisfied(state, context)
                for state, context in zip(states, contexts)
            )
        )

    def termcondition_satisfied(self, state, context=None):
        if self._termcondition_satisfied_fn:
            return self._termcondition_satisfied_fn(self, state, context)
        else:
            raise NotImplementedError

    def termcondition_satisfied_vec(self, states, contexts=None):
        if contexts is None:
            contexts = [None] * len(states)
        return np.array(
            (
                self.termcondition_satisfied(state, context)
                for state, context in zip(states, contexts)
            )
        )

    def make_policy(self, state, context=None):
        """The resulting policy is called externally."""
        pass

    def apply(self, env, obs, context):
        """
        The skill is applied. Also does precondition and termcondition
        checks.
        """
        if self._apply_fn:
            return self._apply_fn(env, obs, context)
        else:
            return NotImplementedError

    @property
    def goal(self):
        """
        Returns: GoalConstraint
        """
        return self.goal_constraint

    @property
    def relevant_cols(self):
        return self.goal.relevant_cols

    # def abstraction(self):
        # return self.goal_constraint.abstraction

    def reset(self, state, context):
        self._start_timestep = context["timestep"]
        return True

    # Training methods
    def train(self, train_cfg):
        if train_cfg['train_policy']:
            self.train_policy(**train_cfg['policy_args'])
        if train_cfg['train_preconds']:
            self.train_precondition(**train_cfg['precond_args'])
        if train_cfg['train_termconds']:
            self.train_termcondition(**train_cfg['termcond_args'])

    def train_policy(self, **kwargs):
        raise NotImplementedError

    def train_precondition(self, **kwargs):
        raise NotImplementedError

    def train_termcondition(self, **kwargs):
        raise NotImplementedError
