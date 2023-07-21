import logging
import pickle as pkl

from hydra.utils import to_absolute_path
import numpy as np
from sklearn.svm import OneClassSVM
from .robot_skill import RobotSkill
from recovery_skills.graph.preconditions import PreconditionClassifier

logger = logging.getLogger(__name__)


class IdentityEESpaceSkill(RobotSkill):
    """
    The policy is to re-execute a end-effector trajectory using impedance
    control.
    """

    def __init__(self, path_to_policy=None, actions=None, timesteps=None):
        super().__init__()

        if path_to_policy:
            self.actions, self.timesteps = pkl.load(open(path_to_policy, 'rb'))
        else:
            self.actions = actions
            if actions is not None and timesteps is None:
                self.timesteps = [1] * len(actions)
            else:
                self.timesteps = timesteps

        # preconds
        self.preconds = None

    def precondition_satisfied(self, state, context=None):
        if self.preconds:
            return self.preconds.is_satisfied(state, context)
        else:
            return True

    # def precondition_satisfied_vec(self, states, contexts=None):
        # if contexts is None:
            # contexts = [None] * len(states)
        # preds = self.preconds.is_satisfied(states)
        # return preds

    def termcondition_satisfied(self, state, context=None):
        return False

    def make_policy(self, state, context=None):
        raise NotImplementedError

    def apply(self, env, obs, context, render=False):
        all_rews = []
        for action, timesteps in zip(self.actions, self.timesteps):
            for _ in range(timesteps):
                obs, rew, done, info = env.step(action)
                all_rews.append(rew)
                if render:
                    env.render()
        info['hist'] = {'all_rews': all_rews}
        return obs, rew, done, info

    def update_policy(self, new_params, context=None):
        self.actions = new_params.reshape(-1, 7)

    def train_policy(self):
        print("Choices:")
        print("1. Load from file")
        choice = int(input("Pick an option: "))
        if choice == 1:
            filepath = input("Enter path to action file: ")
            self.actions, self.timesteps = pkl.load(
                open(to_absolute_path(filepath), "rb")
            )

    def train_precondition(self, states, rl_states, y):
        self.preconds = PreconditionClassifier(states, rl_states, y)
