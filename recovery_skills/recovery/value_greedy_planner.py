from copy import deepcopy
import logging
import numpy as np
import mdptoolbox as mdpt

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class ValueGreedyPlanner():
    """Tracks improvement of recovery using their UCB and picks recovery that
    will most improve the value"""

    def __init__(self, nclusters, nsubgoals, cluster_probs, p_low, c=1,
                 initialize=2, **kwargs):
        self.nsubgoals = nsubgoals
        self.nclusters = nclusters
        self.narms = nsubgoals * nclusters
        self.cluster_probs = cluster_probs
        self.p_low = p_low
        self.reset()

        self.initialize = initialize
        self.c = c
        self.action_rew = -1.0
        self.failure_rew = -5.0

        self.T = self._construct_transition_matrix(p_low)
        self.R = self._construct_reward_matrix()

        logger.info("Initializing value MAB:")
        logger.info(f"  nclusters: {self.nclusters}")
        logger.info(f"  nsubgoals: {self.nsubgoals}")
        logger.info(f"  narms: {self.narms}")
        logger.info(f"  cluster probs: {cluster_probs}")

        self.V = self.compute_value_function()
        self.v_best = self.compute_failure_value()

    def reset(self):
        self.t = 0
        self.dones = [False]*self.narms
        self.p_improvements = [[] for _ in range(self.narms)]
        self.v_best = -np.inf
        self.arm_pulls_hist = []
        self.val_accs = np.zeros(self.narms)

    def replan(self):
        self.t += 1
        arm_pulls = [self.arm_pulls(i) for i in range(self.narms)]
        if min(arm_pulls) < self.initialize:
            best_arm = np.argmin(arm_pulls)
        else:
            improv_rates = self.upper_confidence_bounds()
            predicted_v_improvs = []
            for arm_id, rate in enumerate(improv_rates):
                v_improv = self.compute_potential_value_improv(arm_id, rate)
                predicted_v_improvs.append(v_improv)
            best_arm = np.argmax(predicted_v_improvs)
            max_ucb = max(predicted_v_improvs)
            thresh = 0.001
            best_arms = np.argwhere(predicted_v_improvs > (max_ucb - thresh)).flatten()
            best_arm = np.random.choice(best_arms)
        logger.debug(f"    Arm: {best_arm}")
        self.arm_pulls_hist.append(best_arm)
        cluster_id, subgoal_id = self.get_cluster_subgoal_id(best_arm)
        return cluster_id, subgoal_id

    def compute_potential_value_improv(self, arm_id, improv_rate):
        T = deepcopy(self.T)
        cluster_id, subgoal_id = self.get_cluster_subgoal_id(arm_id)
        start_state_id = self.get_state_id(cluster_id, cluster=True),
        goal_state_id = self.get_state_id(subgoal_id, subgoal=True)
        curr_p = T[start_state_id, goal_state_id]
        predicted_p = min(1.0, curr_p + improv_rate)
        T[1 + subgoal_id, start_state_id, goal_state_id] = predicted_p
        T = self._normalize_transition_matrix(T)
        V = self.compute_value_function(T)
        value = self.compute_failure_value(V)
        value_improv = value - self.v_best
        return value_improv

    def update(self, cluster_id, subgoal_id, p_recovery):
        arm_id = self.get_arm_id(cluster_id, subgoal_id)
        self.update_transition_matrix(p_recovery)
        rew = self.reward()
        logger.debug(f"  Reward: {rew}")
        self.rews[arm_id].append(rew)

    def set_arm_done(self, cluster_id, subgoal_id):
        arm_id = self.get_arm_id(cluster_id, subgoal_id)
        self.dones[arm_id] = True

    def upper_confidence_bounds(self):
        """UCB1"""
        ucbs = []
        n = self.total_arm_pulls
        sis, nis, muis = [], [], []

        for arm_id in range(self.narms):
            si = sum(self.rews[arm_id])
            ni = self.arm_pulls(arm_id)
            if ni > 0:
                mui = si / ni
                ucb = mui + self.c * np.sqrt((2*np.log(n)) / ni)
            else:
                mui = np.nan
                ucb = np.nan
            sis.append(si), nis.append(ni), muis.append(mui)
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

    def compute_failure_value(self, V):
        failure_values = V[self.cluster_ids]
        failure_value = np.dot(failure_values, self.cluster_probs)
        return failure_value

    def compute_value_function(self, T):
        logger.info("  Computing value function for nominal chain under optimistic uncertainty")
        logger.info(f"    action_rew: {self.action_rew}, failure_rew: {self.failure_rew}")

        vi = mdpt.mdp.ValueIteration(T, self.R, 1.0)
        vi.run()
        V = np.array(vi.V)
        logger.debug("    Value Function:")
        logger.debug(f"{V}")
        return V

    def update_transition_matrix(self, cluster_id, subgoal_id, p_recovery):
        start_state_id = self.get_state_id(cluster_id, cluster=True),
        goal_state_id = self.get_state_id(subgoal_id, subgoal=True)
        self.T[1 + subgoal_id, start_state_id, goal_state_id] = p_recovery
        self.T = self._normalize_transition_matrix(self.T)

    def _construct_transition_matrix(self, p_low):
        # two copies of safe states, one of failure states and one final failure
        n, m = self.nsubgoals, self.nclusters
        n_actions = n + 1 #n recovery + 1 nominal skill per state
        n_states = self.num_states
        T = np.zeros((n_actions, n_states, n_states))
        T[0, :, :] = p_low

        # goal
        T[:, self.goal_state_id, :] = 0.0
        T[:, self.goal_state_id, self.goal_state_id] = 1.0

        # failure
        T[:, self.absorbing_failure_id, self.goal_state_id] = 1.0

        T = self._normalize_transition_matrix(T)

        return T

    def _normalize_transition_matrix(self, T):
        # check if nan
        for i in range(len(T[0])):
            if np.isnan(T[0, i, 0]):
                # go to absorbing failure
                T[0, i] = np.zeros(self.num_states)
                T[0, i, -1] = 1.0

        # absorbing failure
        T[:, self.absorbing_failure_id, self.goal_state_id] = 1.0

        for T_action in T:
            success_ps = np.sum(T_action[:, :-1], axis=1)
            T_action[:, -1] = 1 - success_ps

        return T

    def _construct_reward_matrix(self):
        R = np.zeros(self.num_states)
        R[:] = self.action_rew
        # goal
        R[self.goal_state_id] = 0.0
        # failure
        R[-1] = self.failure_rew
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
            return self.nsubgoals + idx
        else:
            return self.absorbing_failure_id

    def get_arm_id(self, cluster_id, subgoal_id):
        return cluster_id * self.nsubgoals + subgoal_id

    def get_cluster_subgoal_id(self, arm_id):
        return arm_id // self.nsubgoals, arm_id % self.nsubgoals

    def arm_pulls(self, arm_id):
        return len(self.rews[arm_id])

    @property
    def total_arm_pulls(self):
        # return sum([len(rews) for rews in self.rews])
        return len(self.arm_pulls_hist)

    @property
    def subgoal_ids(self):
        return np.arange(self.nsubgoals)

    @property
    def cluster_ids(self):
        return np.arange(self.nsubgoals, self.nsubgoals + self.nclusters)

    @property
    def absorbing_failure_id(self):
        return self.nsubgoals + self.nclusters

    @property
    def goal_state_id(self):
        return self.nsubgoals - 1

    @property
    def num_states(self):
        return self.nsubgoals + self.nclusters + 1
