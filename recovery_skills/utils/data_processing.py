import logging
from numbers import Number

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import *
from recovery_skills.recovery.recovery_task import *


logger = logging.getLogger(__name__)



class SelectVarsTransformer(BaseEstimator, TransformerMixin):
    """Selects certain variables from the state."""

    def fit(self, X=None, y=None):
        return self

    def transform(self, states, y=None):
        sub_states = []
        for state in states:
            pos_wrt_handle = state['robot_eef:pose/position'] - state['handle:pose/position']
            handle_theta = state['handle:pose/theta']
            if isinstance(handle_theta, Number):
                handle_theta = [handle_theta]
            sub_state = np.concatenate([pos_wrt_handle, handle_theta])
            sub_states.append(sub_state)
        X = np.array(sub_states)
        return X


def split_based_on_hinge_angle(states):
    hinges = np.array([state['hinge:pose/theta'] for state in states])
    states = np.array(states)

    thresh = 0.05
    door_closed = states[hinges <= thresh]
    door_open = states[hinges > thresh]

    # open = hinges[hinges >= thresh]
    # plt.scatter(np.arange(len(open)), open)
    # plt.show()

    return door_closed, door_open


def failures_to_tasks(all_failures, nominal_skill_chain, subgoals):
    """Convert failures into recovery tasks."""
    nclusters = len(all_failures)
    skills = nominal_skill_chain.skills

    tasks = [[] for _ in range(nclusters)]

    for cluster_id, failures in enumerate(all_failures):
        for failure in failures:
            start = failure
            tasks[cluster_id].append(
                [
                    RecoveryTask(start, goal, goal_id)
                    for goal_id, goal in enumerate(subgoals)
                ]
            )

    return tasks


def split_into_train_and_val_tasks(tasks, cluster_ids, nclusters, nsubgoals,
                                   iters=False):
    # split into train and val
    tasks_train = [[] for _ in range(nclusters)]
    tasks_val = [[] for _ in range(nclusters)]

    for cluster_id in cluster_ids:
        logger.info(f"  Cluster: {cluster_id}")
        all_tasks = np.array(tasks[cluster_id])
        for goal_id in range(nsubgoals):
            logger.info(f"    goal_id: {goal_id}")
            if len(all_tasks) >= 10:
                _train, _val = train_test_split(all_tasks[:, goal_id],
                                                shuffle=True,
                                                test_size=0.2)
            else:
                _train, _val = [], []
            logger.info(f"      train : val :: {len(_train)} : {len(_val)}")
            if iters:
                tasks_train[cluster_id].append(iter(_train))
                tasks_val[cluster_id].append(iter(_val))
            else:
                tasks_train[cluster_id].append(list(_train))
                tasks_val[cluster_id].append(list(_val))

    return tasks_train, tasks_val

