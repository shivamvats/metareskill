import logging
import numpy as np
from scipy.stats import beta

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class MetaMABPlanner():
    """Manages MABs for all clusters to choose both the cluster and arm."""

    def __init__(self, mabs):
        self.mabs = mabs
        self.nclusters = len(mabs)
        self.t = 0
        self.ntasks = mabs[0].ntasks
        self.dones = [False]*len(mabs)

    def step(self):
        # Assume equal sized clusters
        cluster_ids = np.arange(self.nclusters)
        remaining_cluster_ids = cluster_ids[np.logical_not(self.dones)]
        cluster_id = np.random.choice(remaining_cluster_ids)
        mab = self.mabs[cluster_id]
        arm_id = mab.pull()

        return cluster_id, arm_id

    def update(self, cluster_id, subgoal_id, reward):
        done = self.mabs[cluster_id].update(subgoal_id, reward)
        # self.dones[cluster_id] = done
        return all(self.dones)

    def reset(self):
        self.t = 0
        self.dones = [False]*self.nclusters
        for mab in self.mabs:
            mab.reset()

    def remaining_tasks(self):
        return np.vstack([mab.nremaining_tasks for mab in self.mabs])

    def solved_tasks(self):
        return np.vstack([self.ntasks - np.array(mab.nremaining_tasks) for mab in self.mabs])


class MABPlanner:
    """Implements a Beta multi-armed bandit-based strategy for solving the recovery
    tasks.

    Creates an arm for each subgoal. Maintains UCB estimates of
    probability of success of solving a recovery task to each subgoal.

    Picks the arm that is expected to finish early--- complete all recovery
    tasks the quickest.
    """

    def __init__(self, narms, ntasks=np.inf, init_a=1, init_b=1, c=2, initialize=0):
        self.narms = narms
        self.ntasks = ntasks
        self.prior = (init_a, init_b)
        # num of stds to use in ucb
        self.c = c
        self.initialize = initialize
        self.reset()

    def reset(self):
        # prior is a beta(a, b) distribution
        self._as = [self.prior[0]] * self.narms
        self._bs = [self.prior[1]] * self.narms
        self.nremaining_tasks = [self.ntasks for _ in range(self.narms)]
        self.t = 0
        self.arm_pulls = []
        self.arm_dones = [False]*self.narms

    def pull(self):
        arm_pulls = np.array([a + b for a, b in zip(self._as, self._bs)])
        arm_pulls = arm_pulls - np.sum(self.prior)

        if min(arm_pulls) < self.initialize:
            best_arm = np.argmin(arm_pulls)

        else:
            probs = self.expected_probs()
            logger.debug(f"  Mean: {probs}")

            p_ucbs = np.array(self.upper_confidence_bounds())
            # these arms are done
            p_ucbs[self.arm_dones] = -1

            logger.debug(f"  UCB: {p_ucbs}")

            best_arm = np.argmax(p_ucbs)
            max_ucb = max(p_ucbs)
            thresh = 0.001
            best_arms = np.argwhere(p_ucbs > (max_ucb - thresh)).flatten()
            best_arm = np.random.choice(best_arms)

            # compute expected time to complete
            # time_to_complete = np.array([
                # ntasks / p for ntasks, p in zip(self.nremaining_tasks, p_ucbs)
            # ])
            # min_time_to_complete = np.min(time_to_complete)

            # thresh = 0.001
            # best_arms = np.argwhere(time_to_complete < (min_time_to_complete +
                                                        # thresh)).flatten()
            # best_arm = np.random.choice(best_arms)
            # # if best_arm == 0:
                # # __import__('ipdb').set_trace()
            # logger.debug(f"  Time to complete: {time_to_complete}")

        # logger.debug(f"  Best Arm: {best_arm}")
        self.arm_pulls.append(best_arm)
        return best_arm

    def update(self, arm_id, reward):
        """Each update is a bernoulli trial"""
        assert reward in [0, 1]

        logger.debug(f"  Reward: {reward}")

        self.t += 1

        # posterior is also a beta distribution
        if reward:
            self.nremaining_tasks[arm_id] -= 1

        self._as[arm_id] += reward
        self._bs[arm_id] += (1 - reward)

        done = min(self.nremaining_tasks) == 0

        return done

    def expected_probs(self):
        expected_probs = [
            self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.narms)
        ]
        return expected_probs

    def upper_confidence_bounds(self):
        ucbs = []
        for i in range(self.narms):
            expected_val = self._as[i] / (self._as[i] + self._bs[i])
            std = beta.std(self._as[i], self._bs[i])
            ucb = expected_val + self.c * std
            ucbs.append(ucb)
        return ucbs

    def info_dict(self):
        pos_samples = np.array(self._as) - self.prior[0]
        neg_samples = np.array(self._bs) - self.prior[1]
        total_samples = pos_samples + neg_samples
        info = {
            'positive_samples': pos_samples,
            'negative_samples': neg_samples,
            'total_samples': total_samples,
        }
        return info

