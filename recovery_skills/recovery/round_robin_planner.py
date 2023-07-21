import math
import numpy as np
from itertools import cycle


class RoundRobinPlanner():
    def __init__(self, nclusters, nsubgoals, budget, weights=None):
        self.nclusters = nclusters
        self.nsubgoals = nsubgoals
        self.budget = budget
        self.weights = weights
        self.narms = nclusters * nsubgoals
        self.narm_pulls = np.zeros(self.narms)

        self.reset()

    def reset(self):
        self.dones = np.zeros(self.narms)
        self.arm_pulls_hist = []
        self.iter = 0
        self.dones = np.zeros((self.nclusters, self.nsubgoals), dtype=bool)

        if self.weights is None:
            weights = np.ones(self.nclusters)
            weights /= np.sum(weights)
            ns = [math.ceil(weight * self.budget) for weight in weights]

        else:
            ns = [math.ceil(weight * self.budget) for weight in self.weights]
            allocated_budget = np.sum(ns)
            while allocated_budget > self.budget:
                # sample index in proprtion to size
                sampled_id = np.random.choice(np.arange(self.nclusters), p=self.weights)
                ns[sampled_id] -= 1
                allocated_budget = np.sum(ns)
        self.cluster_budgets = ns
        self.cluster_cycles = [cycle(range(self.nsubgoals)) for _ in
                               range(self.nclusters)]

    def pull(self):
        if not any(self.cluster_budgets):
            raise ValueError("Out of budget")

        next_cluster_id, next_subgoal_id = 0, 0
        for cluster_id, budget in enumerate(self.cluster_budgets):
            if all(self.dones[cluster_id]):
                continue

            if budget > 0:
                while True:
                    next_subgoal_id = next(self.cluster_cycles[cluster_id])
                    if not self.dones[cluster_id][next_subgoal_id]:
                        break
                next_cluster_id = cluster_id
                break
        self.cluster_budgets[next_cluster_id] -= 1
        arm_id  = self.get_arm_id(next_cluster_id, next_subgoal_id)
        self.arm_pulls_hist.append(arm_id)
        self.narm_pulls[arm_id] += 1
        return next_cluster_id, next_subgoal_id

    def get_arm_id(self, cluster_id, subgoal_id):
        return cluster_id * self.nsubgoals + subgoal_id

    def set_arm_done(self, cluster_id, subgoal_id):
        self.dones[cluster_id][subgoal_id] = True

    def arm_pulls(self, arm_id):
        return self.narm_pulls[arm_id]

    def update(self, cluster_id, subgoal_id, *args):
        pass

    def info_dict(self):
        pass
