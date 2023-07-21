from copy import deepcopy
import logging
import numpy as np
import scipy.stats as stats

from .value_mab_planner import MonoValueMABPlanner

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
# logger.setLevel("INFO")


class TransitionUCBPlanner(MonoValueMABPlanner):
    """Maintains UCB estimates on rate of improvement of each recovery policy
    separately. Then picks the policy for training that will result in the
    largest increase in value of the start."""

    def __init__(self, sym_graph, p_trans, fail_weights=None, c=1, initialize=2, discount=1,
                 planning_horizon=1, window=0, **kwargs):
        self.fail_weights = deepcopy(fail_weights)
        self.planning_horizon = planning_horizon
        self.window = window
        super().__init__(sym_graph, None, p_trans, c, initialize, discount,
                         **kwargs)

    def reset(self):
        self.t = 0
        self.dones = [False]*self.narms
        self.narm_pulls = np.zeros(self.narms)
        self.arm_pulls_hist = []
        self.v_best = np.inf
        self.p_recovery_hist = [[0.0] for _ in range(self.narms)]
        self.info = {'hist': {
            'p_recovery': [np.zeros(self.narms)],
            'V': []
        }}

    def pull(self):
        self.t += 1
        arm_pulls = np.array([self.arm_pulls(i) for i in range(self.narms)])
        T_cache = deepcopy(self.T)
        # don't pull done arms
        arm_pulls[self.dones] = 10**5
        if min(arm_pulls) < self.initialize:
            best_arm = np.argmin(arm_pulls)
        else:
            del_p_ucbs = self.upper_confidence_bounds()
            # avoid done arms
            del_p_ucbs[self.dones] = -1.0
            # UCB Estimate
            p_recovery = self.p_recovery()
            p_ucbs = p_recovery + del_p_ucbs * self.planning_horizon
            # Compute value with each UCB estimate
            v_ucbs = []
            for i, p_ucb in enumerate(p_ucbs):
                p_potential = deepcopy(p_recovery)
                # Ensure bound on probability
                p_potential[i] = np.clip(p_ucb, 0.0, 1.0)
                # compute value with this p mat
                p_potential = np.reshape(p_potential,
                                        (self.sym_graph.num_fail_modes,
                                        self.sym_graph.num_subgoals))
                self.update_transition_matrix(p_potential)
                try:
                    pi, V = self.solve_mdp()
                except:
                    __import__('ipdb').set_trace()

                if self.fail_weights is None:
                    # value of start
                    v_ucbs.append(V[0])
                else:
                    fail_value = self.compute_failure_value(V)
                    v_ucbs.append(fail_value)

            best_arm = np.argmax(v_ucbs)
            max_ucb = max(v_ucbs)
            thresh = 0.001
            best_arms = np.argwhere(v_ucbs > (max_ucb - thresh)).flatten()
            best_arm = np.random.choice(best_arms)
        # reset transition matrix
        self.T = T_cache
        logger.debug(f"    Arm: {best_arm}")
        self.arm_pulls_hist.append(best_arm)
        self.narm_pulls[best_arm] += 1
        cluster_id, subgoal_id = self.get_cluster_subgoal_id(best_arm)
        return cluster_id, subgoal_id

    def update(self, cluster_id, subgoal_id, p_recovery):
        self.info['hist']['p_recovery'].append(deepcopy(p_recovery))
        arm_id = self.get_arm_id(cluster_id, subgoal_id)
        self.p_recovery_hist[arm_id].append(p_recovery[cluster_id, subgoal_id])
        self.update_transition_matrix(p_recovery)
        pi, V = self.solve_mdp()
        if self.fail_weights is None:
            # value of start
            v = V[0]
            self.info['hist']['V'].append(v)
        else:
            fail_value = self.compute_failure_value(V)
            v = fail_value
        self.v_best = v

    def compute_failure_value(self, V=None):
        if V is None:
            V = self.V
        failure_values = V[self.sym_graph.fail_mode_ids]
        failure_value = np.dot(failure_values, self.fail_weights)
        return failure_value

    def p_recovery(self):
        p_recovery = []
        for fail_id in self.sym_graph.fail_mode_ids:
            for i, subgoal_id in enumerate(self.sym_graph.subgoal_ids):
                p = self.T[1 + i][fail_id][subgoal_id]
                p_recovery.append(p)
        return np.array(p_recovery)

    def del_ps(self):
        delta_p_recovery = self.p_recovery() # - self.p_trans_init
        arm_pulls = np.array([self.arm_pulls(i) for i in range(self.narms)])
        del_ps = delta_p_recovery / arm_pulls
        del_ps = np.nan_to_num(del_ps)
        return del_ps

    def upper_confidence_bounds(self):
        """UCB1"""
        ucbs = []
        n = self.total_arm_pulls
        del_ps = self.del_ps()

        # if self.discount == 1:
        if self.window > 0:
            # compute confidence interval
            for arm_id in range(self.narms):
                ni = self.arm_pulls(arm_id)

                if ni > 0:
                    ps = self.p_recovery_hist[arm_id]
                    window_ps = np.array(ps[-(self.window + 1):])
                    del_ps = []
                    for i in range(1, len(window_ps)):
                        del_ps.append(window_ps[i] - window_ps[i-1])
                    mean = np.mean(del_ps)
                    std = stats.sem(del_ps)
                    conf_interval = stats.t.interval(alpha=0.95,
                                                     df=len(del_ps) - 1,
                                                     loc=mean,
                                                     scale=std)

                    uci = conf_interval[1]
                    uci = np.nan_to_num(uci)
                else:
                    uci = 0.5
                ucbs.append(uci)

        else:
            for arm_id in range(self.narms):
                ni = self.arm_pulls(arm_id)
                if ni > 0:
                    mui = del_ps[arm_id]
                    ucb = mui + self.c * np.sqrt((2*np.log(n)) / ni)
                    # UCB-tuned
                    # vari = np.var(self.del_ps[arm_id])
                    # ucb = mui + self.c * np.sqrt((np.log(n) / ni) *
                                                 # min(0.25, vari + 2*np.log(n)/ni))
                else:
                    ucb = 0.1
                ucbs.append(ucb)

        arm_pulls = [self.arm_pulls(i) for i in range(self.narms)]
        # logger.debug(f"   Mean: {np.round(mean_del_ps, 2)}")
        logger.debug(f"  Arm pulls: {arm_pulls}")
        logger.debug(f"  UCB: {np.round(ucbs, 2)}")
        ucbs = np.array(ucbs)
        return ucbs

    def info_dict(self):
        dones = self.dones
        del_ps = self.del_ps()
        v_best = self.v_best
        arm_pulls_hist = self.arm_pulls_hist
        info = {
            'dones': dones,
            'del_ps': del_ps,
            'v_best': v_best,
            'arm_pulls_hist': arm_pulls_hist,
            'arm_pulls': [self.arm_pulls(i) for i in range(self.narms)],
            'total_arm_pulls': self.total_arm_pulls,
            'hist': self.info['hist']
        }
        return info

    @property
    def rews(self):
        return self.del_ps()
