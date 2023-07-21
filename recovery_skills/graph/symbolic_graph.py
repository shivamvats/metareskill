import logging

import numpy as np
import mdptoolbox as mdpt

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

class SymbolicGraph():

    def __init__(self, preconds, failure_clf):
        self.preconds = preconds
        self.failure_clf = failure_clf

        self.num_subgoals = len(preconds)
        self.num_fail_modes = sum(failure_clf.nclusters)

        self.state_costs = np.ones(self.num_states)
        self.state_rews = -1 * self.state_costs
        self.failure_rew = -10.0

        self.T = None
        self.R = None
        self.pi = None
        self.V = None

    def construct_reward_matrix(self):
        R = np.zeros(self.num_states)
        R[:] = self.state_rews
        # goal
        R[self.goal_state_id] = 0.0
        # failure
        R[self.absorb_fail_id] = self.failure_rew
        return R

    def solve(self, T=None, R=None, discount=1.0): # horizon=10):
        """Solve given discrete MDP using Value Iteration"""

        if T is None:
            T = self.T
        if R is None:
            R = self.R
        # if horizon:
            # logger.debug("  Solving finite horizon MDP")
            # fh = mdpt.mdp.FiniteHorizon(T, R, discount=discount, N=horizon)
            # fh.run()
            # pi = np.array(fh.policy[-1])
            # V = np.array(fh.V[-1])
            # __import__('ipdb').set_trace()
        # else:
            # logger.debug("  Solving infinite horizon MDP")
        vi = mdpt.mdp.ValueIteration(T, R, discount)
        vi.run()
        pi = np.array(vi.policy)
        V = np.array(vi.V)
        self.update_policy(pi, V)
        return pi, V

    def update_policy(self, pi, V):
        self.pi = pi
        self.V = V

    def state_to_id(self, state):
        for i, precond in enumerate(self.preconds):
            if precond.is_satisfied(state):
                return i

        failure_id = self.failure_clf.predict(state)
        if failure_id == -1:
            return self.absorb_fail_id
        else:
            return self.num_subgoals + failure_id

    def is_subgoal(self, state=None, state_id=None):
        if state_id is None:
            state_id = self.state_to_id(state)
        return state_id in self.subgoal_ids

    def is_fail_mode(self, state=None, state_id=None):
        if state_id is None:
            state_id = self.state_to_id(state)
        return state_id in self.fail_mode_ids

    @property
    def num_states(self):
        return self.num_subgoals + self.num_fail_modes + 1

    @property
    def num_actions(self):
        return self.num_subgoals + 1

    @property
    def absorb_fail_id(self):
        return self.num_subgoals + self.num_fail_modes

    @property
    def goal_state_id(self):
        return self.num_subgoals - 1

    @property
    def subgoal_ids(self):
        return np.arange(self.num_subgoals)

    @property
    def fail_mode_ids(self):
        return np.arange(self.num_subgoals,
                         self.num_subgoals + self.num_fail_modes)

    @property
    def policy(self):
        return self.pi

    def _normalize_transition_matrix(self, T):
        J, I, _ = T.shape

        # check if row is nan
        for j in range(J):
            for i in range(I):
                if np.isnan(T[j, i, 0]):
                    # go to absorbing failure
                    T[j, i] = np.zeros(self.num_states)
                    if i not in [self.absorb_fail_id, self.goal_state_id]:
                        T[j, i, self.absorb_fail_id] = 1.0

        # absorbing failure
        T[:, self.absorb_fail_id, :] = 0.0
        T[:, self.absorb_fail_id, self.goal_state_id] = 1.0

        # absorbing goal
        T[:, self.goal_state_id, :] = 0.0
        T[:, self.goal_state_id, self.goal_state_id] = 1.0

        return T


def estimate_transition_matrix(
        env,
        graph,
        skill_chain,
        recovery_skills,
        nevals,
        MAX_ACTIONS=10,
        render=False
):
    all_info = {'n_evals': 0,
                'n_fails': 0,
                'n_actions': [],
                'success': [],
                'mean_reward': 0,
                'recovery': {
                    'times_triggered': 0,
                    'success': [],
                    'hist':[],
                    'eval_hist': [[] for _ in range(nevals)],
                }
            }
    T = np.zeros((graph.num_actions, graph.num_states, graph.num_states))

    for i in range(nevals):
        logger.debug(f"Eval {i}")
        env.reset()
        obs = env.obs()
        n_actions = 0
        while True:
            logger.info(f"  Attempt {n_actions}")
            if n_actions >= MAX_ACTIONS or env.check_task_success():
                if not env.check_task_success():
                    logger.info("Max actions exceeded\n")
                break

            start_id = graph.state_to_id(env.obs())

            if graph.is_subgoal(state_id=start_id):
                logger.info("  Nominal skill sat")
                action_id = 0
                obs, rew, done, info = skill_chain.apply_skill(env,
                                                        obs,
                                                        render=render)

            elif graph.is_fail_mode(state_id=start_id):
                logger.info("  Failure mode")
                logger.info("  Executing recovery skill")
                failure_id = graph.failure_clf.predict(obs)
                # learnt skills
                available_action_ids = np.argwhere(recovery_skills[failure_id]).flatten()
                if len(available_action_ids) == 0:
                    break

                recovery_id = np.random.choice(available_action_ids)
                action_id = recovery_id + 1
                all_info['recovery']['hist'].append([failure_id,
                                                    recovery_id])
                all_info['recovery']['eval_hist'][i].append([failure_id,
                                                    recovery_id])
                best_recovery_skill = recovery_skills[failure_id][recovery_id]['skill']
                best_succ_prob = recovery_skills[failure_id][recovery_id]['accuracy']
                logger.info(f"    Skill to subgoal {recovery_id} with success prob: {best_succ_prob}")

                all_info['recovery']['times_triggered'] += 1
                obs, rew, done, info = best_recovery_skill.apply(env,
                                                            obs,
                                                            render=render)
            else:
                logger.info("  Failure state. Breaking")
                all_info['recovery']['hist'].append([failure_id,
                                                    -1])
                all_info['recovery']['success'].append(False)
                T[action_id, start_id, graph.absorb_fail_id] += 1
                break

            #halve uncertainty
            obs = env.obs(force_update=True)
            term_id = graph.state_to_id(obs)
            T[action_id, start_id, term_id] += 1

            n_actions += 1

    p_transition = T / np.sum(T, axis=2, keepdims=True)

    return p_transition, T


def construct_optimistic_transition_mat(sym_graph):
    p_trans = np.zeros((sym_graph.num_states, sym_graph.num_states))
    n_nominal_skills = sym_graph.num_subgoals - 1
    for i in range(n_nominal_skills):
        p_trans[i, i+1] = 1.0
    p_trans[sym_graph.goal_state_id, sym_graph.goal_state_id] = 1.0
    return p_trans
