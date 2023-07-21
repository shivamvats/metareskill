from copy import deepcopy
import logging
import numpy as np

from recovery_skills.utils import *

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
# logger.setLevel("INFO")


class MonoValueMABPlanner():
    """Implements the UCB1 algorithm for solving a multi-armed bandit
    problem."""

    def __init__(self, sym_graph, cluster_probs, p_trans, c=1,
                 initialize=2, discount=1, **kwargs):
        self.sym_graph = sym_graph
        self.narms = sym_graph.num_subgoals * sym_graph.num_fail_modes
        self.cluster_probs = cluster_probs
        self.p_trans_init = p_trans
        self.discount = discount
        self.reset()

        self.initialize = initialize
        self.c = c

        self.state_costs = np.ones(self.sym_graph.num_states)
        self.state_rews = -1 * self.state_costs
        self.failure_rew = -10.0

        # TODO Move to symbolic graph
        self.T = self._construct_transition_matrix(p_trans)
        self.R = self._construct_reward_matrix()

        logger.debug("Initializing value MAB:")
        logger.debug(f"  nclusters: {self.sym_graph.num_fail_modes}")
        logger.debug(f"  nsubgoals: {self.sym_graph.num_subgoals}")
        logger.debug(f"  narms: {self.narms}")
        logger.debug(f"  cluster probs: {cluster_probs}")

        self.policy, self.V = self.solve_mdp()
        if not self.cluster_probs is None:
            self.v_best = self.compute_failure_value()

    def reset(self):
        self.t = 0
        self.dones = [False]*self.narms
        self.rews = [[] for _ in range(self.narms)]
        self.v_best = -np.inf
        self.arm_pulls_hist = []
        self.narm_pulls = np.zeros(self.narms)

    def pull(self):
        self.t += 1
        arm_pulls = np.array([self.arm_pulls(i) for i in range(self.narms)])
        # don't pull done arms
        arm_pulls[self.dones] = 10**5
        if min(arm_pulls) < self.initialize:
            best_arm = np.argmin(arm_pulls)
        else:
            ucbs = self.upper_confidence_bounds()
            ucbs[self.dones] = -np.inf
            best_arm = np.argmax(ucbs)
            max_ucb = max(ucbs)
            thresh = 0.001
            best_arms = np.argwhere(ucbs > (max_ucb - thresh)).flatten()
            best_arm = np.random.choice(best_arms)
        logger.debug(f"    Arm: {best_arm}")
        self.arm_pulls_hist.append(best_arm)
        self.narm_pulls[best_arm] += 1
        cluster_id, subgoal_id = self.get_cluster_subgoal_id(best_arm)
        return cluster_id, subgoal_id

    def update(self, cluster_id, subgoal_id, p_recovery):
        arm_id = self.get_arm_id(cluster_id, subgoal_id)
        self.update_transition_matrix(p_recovery)
        rew = self.reward()
        logger.debug(f"  Reward: {rew}")
        self.rews[arm_id].append(rew)

    def set_arm_done(self, cluster_id, subgoal_id):
        arm_id = self.get_arm_id(cluster_id, subgoal_id)
        self.dones[arm_id] = True

    # def expected_rews(self):
        # pass
        # expected_probs = [
            # self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.narms)
        # ]
        # return expected_probs

    def upper_confidence_bounds(self):
        """UCB1"""
        ucbs = []
        n = self.total_arm_pulls
        sis, nis, muis = [], [], []

        if self.discount == 1:
            for arm_id in range(self.narms):
                si = sum(self.rews[arm_id])
                ni = self.arm_pulls(arm_id)
                if ni > 0:
                    mui = si / ni
                    ucb = mui + self.c * np.sqrt((2*np.log(n)) / ni)
                else:
                    mui = 0.0
                    ucb = 0.5
                sis.append(si), nis.append(ni), muis.append(mui)
                ucbs.append(ucb)

        else:
            discounts = [1.0]
            for _ in range(n):
                discount = discounts[-1]*self.discount
                discounts.append(discount)
            discounts = discounts[::-1]

            for arm_id in range(self.narms):
                arm_discounts = []
                for i in range(n):
                    if self.arm_pulls_hist[i] == arm_id:
                        arm_discounts.append(discounts[i])
                si = np.dot(self.rews[arm_id], arm_discounts)
                ni = sum(arm_discounts)
                sis.append(si), nis.append(ni)
                mui = si / ni
                muis.append(mui)

            nt = sum(nis)

            for arm_id in range(self.narms):
                ucb = muis[arm_id] + self.c * np.sqrt((2*np.log(nt)) / nis[arm_id])
                ucbs.append(ucb)

        mean_rews = [np.mean(rews) for rews in self.rews]
        arm_pulls = [self.arm_pulls(i) for i in range(self.narms)]
        logger.debug(f"   Mean: {np.round(mean_rews, 2)}")
        logger.debug(f"  Discounted Mean: {np.round(muis, 2)}")
        logger.debug(f"  Arm pulls: {arm_pulls}")
        logger.debug(f"  UCB: {np.round(ucbs, 2)}")
        ucbs = np.array(ucbs)
        return ucbs

    # Reward Function
    #   Value Function
    #------------------
    def reward(self):
        _, self.V = self.solve_mdp()
        new_value = self.compute_failure_value()
        reward = new_value - self.v_best
        if new_value > self.v_best:
            logger.debug(f"    Updated best value to {new_value} from {self.v_best}")
            self.v_best = new_value
        return reward

    def compute_failure_value(self, V=None):
        if V is None:
            V = self.V
        failure_values = V[self.sym_graph.fail_mode_ids]
        failure_value = np.dot(failure_values, self.cluster_probs)
        return failure_value

    def solve_mdp(self):
        logger.debug(f"    state_rews: {self.state_rews}, failure_rew: {self.failure_rew}")
        pkl_dump((self.T, self.R), "solving_T_R.pkl")
        pi, V = self.sym_graph.solve(self.T, self.R)
        logger.debug("    Value Function:")
        logger.debug(f"{V}")
        logger.debug("    Policy")
        logger.debug(f"{pi}")
        return pi, V

    def update_transition_matrix(self, p_recovery):
        for i in range(self.sym_graph.num_fail_modes):
            for j in range(self.sym_graph.num_subgoals):
                # jth recovery is from cluster i to subgoal j
                start_state_id = self.get_state_id(i, cluster=True),
                goal_state_id = self.get_state_id(j, subgoal=True)
                self.T[1 + j, start_state_id, goal_state_id] = p_recovery[i, j]
        self.T = self._normalize_transition_matrix(self.T)

    def _construct_transition_matrix(self, p_trans):
        # two copies of safe states, one of failure states and one final failure
        n, m = self.sym_graph.num_subgoals, self.sym_graph.num_fail_modes
        n_actions = n + 1 #n recovery + 1 nominal skill per state
        n_states = self.sym_graph.num_states
        T = np.zeros((n_actions, n_states, n_states))
        # T[0, :, :] = p_trans
        # ignore transitions from failure to other failures
        T[0, :n, :] = p_trans[:n ,:]

        T = self._normalize_transition_matrix(T)

        return T

    def _normalize_transition_matrix(self, T):
        J, I, _ = T.shape

        # check if row is nan
        for j in range(J):
            for i in range(I):
                if np.isnan(T[j, i, 0]):
                    # go to absorbing failure
                    T[j, i] = np.zeros(self.sym_graph.num_states)
                    if i not in [self.sym_graph.absorb_fail_id, self.sym_graph.goal_state_id]:
                        T[j, i, self.sym_graph.absorb_fail_id] = 1.0
        # replace nan with 0
        T = np.nan_to_num(T)

        # absorbing failure
        T[:, self.sym_graph.absorb_fail_id, :] = 0.0
        T[:, self.sym_graph.absorb_fail_id, self.sym_graph.goal_state_id] = 1.0

        # absorbing goal
        T[:, self.sym_graph.goal_state_id, :] = 0.0
        T[:, self.sym_graph.goal_state_id, self.sym_graph.goal_state_id] = 1.0

        for T_action in T:
            success_ps = np.sum(T_action[:, :-1], axis=1)
            T_action[:, -1] = 1 - success_ps

        return T

    def _construct_reward_matrix(self):
        R = np.zeros(self.sym_graph.num_states)
        R[:] = self.state_rews
        # goal
        R[self.sym_graph.goal_state_id] = 0.0
        # failure
        R[self.sym_graph.absorb_fail_id] = self.failure_rew
        return R

    def info_dict(self):
        dones = self.dones
        rews = self.rews
        v_best = self.v_best
        arm_pulls_hist = self.arm_pulls_hist
        info = {
            'dones': dones,
            'rews': rews,
            'v_best': v_best,
            'arm_pulls_hist': arm_pulls_hist,
            'arm_pulls': [self.arm_pulls(i) for i in range(self.narms)],
            'total_arm_pulls': self.total_arm_pulls
        }
        return info

    def get_state_id(self, idx, subgoal=False, cluster=False):
        if subgoal:
            return idx
        elif cluster:
            return self.sym_graph.num_subgoals + idx
        else:
            return self.sym_graph.absorb_fail_id

    def get_arm_id(self, cluster_id, subgoal_id):
        return cluster_id * self.sym_graph.num_subgoals + subgoal_id

    def get_cluster_subgoal_id(self, arm_id):
        return arm_id // self.sym_graph.num_subgoals, arm_id % self.sym_graph.num_subgoals

    def arm_pulls(self, arm_id):
        return self.narm_pulls[arm_id]

    @property
    def total_arm_pulls(self):
        return len(self.arm_pulls_hist)


class ValuePlusMABPlanner():
    """Implements the UCB1-Normal algorithm for solving a multi-armed bandit
    problem.

    Also reasons about the structure of the failure discovery graph.
    """

    def __init__(self, nclusters, nsubgoals, p_trans, p_high, **kwargs):
        self.nsubgoals = nsubgoals
        self.nclusters = nclusters
        self.narms = nsubgoals * nclusters
        # low uncertianty
        self.p_trans = p_trans
        # high uncertianty
        self.p_high = p_high

        self.T = self._construct_transition_matrix(p_high, p_trans)

        logger.debug("Initializing value MAB:")
        logger.debug(f"  nclusters: {self.nclusters}")
        logger.debug(f"  nsubgoals: {self.nsubgoals}")
        logger.debug(f"  narms: {self.narms}")

        self.c = 1 # std
        self.reset()

        self.action_rew = -1.0
        self.failure_rew = -2.0
        self.V = self.compute_value_function()


    def compute_value_function(self):
        """Compute the value function independent of other recoveries."""

        logger.debug("  Computing value function for nominal chain under optimistic uncertainty")
        logger.debug(f"    action_rew: {self.action_rew}, failure_rew: {self.failure_rew}")

        V = np.zeros(self.nsubgoals)
        # first four states are safe states
        # start from the goal
        V[3] = 0
        for i in range(self.nsubgoals-1)[::-1]:
            p_succ = self.p_trans[i, i+1]
            p_fail = sum(self.p_trans[i, self.nsubgoals:])
            V[i] = self.action_rew + p_succ*V[i+1] + self.failure_rew*p_fail
        logger.debug(f"    V: {V}")
        return V

    def step(self):
        arm_id = 0
        return self.get_cluster_subgoal_id(arm_id)

    def update(self, cluster_id, subgoal_id, p):
        pass

    def expected_rews(self):
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

    def set_arm_done(self, cluster_id, subgoal_id):
        arm_id = self.get_arm_id(cluster_id, subgoal_id)
        self.dones[arm_id] = True

    def reset(self):
        self.t = 0
        self.dones = [False]*self.narms
        self.rews = []
        self.arm_pulls = np.zeros(self.narms, int)

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

    def get_arm_id(self, cluster_id, subgoal_id):
        return cluster_id * self.nsubgoals + subgoal_id

    def get_cluster_subgoal_id(self, arm_id):
        return arm_id // self.nsubgoals, arm_id % self.nsubgoals
