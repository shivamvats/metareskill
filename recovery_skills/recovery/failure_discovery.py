from copy import deepcopy
import logging
import numpy as np

from recovery_skills.graph.symbolic_graph import *

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

def discover_failures(env, skill_chain, recoveries, planner,
                      preconds, failure_clf,
                      nfails, pessimistic_discovery=True, early_term=False,
                      MAX_ACTIONS=10, render=False):

    if recoveries is None:
        recover = False
        openloop_discovery = True
        logger.debug("Open-loop failure discovery")
    elif early_term:
        recover = False
        openloop_discovery = False
        logger.debug("Early termination failure discovery")
    else:
        recover = True
        openloop_discovery = False
        logger.debug("Failure discovery with recoveries")


    nsubgoals = len(preconds)
    nfailure_clusters = sum(failure_clf.nclusters)
    graph = SymbolicGraph(preconds, failure_clf)
    nstates = nsubgoals + nfailure_clusters + 1 # subgoal, failure clustes, FAIL
    n_actions = len(skill_chain.skills)

    if planner:
        policy, V = planner.solve_mdp()

    T_nominal = np.zeros((nstates, nstates))
    fail_states = []
    all_info = {'n_evals': 0,
                'n_fails': 0,
                'success': [],
                'rewards': [],
                'mean_reward': 0,
                'recovery': {
                    'times_triggered': 0,
                    'success': [],
                    'hist':[],
                    'eval_hist': [],
                }
            }

    for i in range(2000):
        if len(fail_states) >= nfails:
            break

        logger.debug(f"Evaluation {i}")
        logger.debug("------------")
        env.set_obs_corruption(True)
        env.reset()
        obs = env.obs()
        all_info['n_evals'] += 1
        all_info['T_nominal'] = T_nominal
        all_info['recovery']['eval_hist'].append([])
        n_actions = 0

        if openloop_discovery or early_term:
            for i, skill in enumerate(skill_chain.skills):
                # gt_obs = env.obs(ground_truth=True)
                obs = env.obs()
                start_id = graph.state_to_id(obs)
                env.set_goal_constraint(skill.goal)
                obs, rew, done, info = skill.apply(env, obs)

                if not pessimistic_discovery:
                    obs = env.obs(force_update=True)

                obs = env.obs()
                term_id = graph.state_to_id(obs)

                success = preconds[i+1].is_satisfied(obs)
                task_success = env.check_task_success()
                if task_success:
                    term_id = graph.goal_state_id
                    success = True

                T_nominal[start_id, term_id] += 1

                # ground truth
                # I only care about ground truth failure states
                gt_obs = env.obs(ground_truth=True)
                if not skill_chain.precondition_satisfied(gt_obs):
                    failure = gt_obs
                    logger.debug("Failure state found")
                    fail_states.append(failure)

                # obs
                if success:
                    all_info['success'].append(True)

                if not success:
                    info = {"is_solved": False}
                    all_info['success'].append(False)
                    if early_term:
                        # recovery assumed to succeed
                        break

                if task_success:
                    break

        else:
            while True:
                logger.debug(f"Attempt {n_actions}")
                if n_actions >= MAX_ACTIONS or env.check_task_success():
                    if not env.check_task_success():
                        logger.debug("Max actions exceeded\n")
                    break
                # start_id = graph.state_to_id(env.obs(ground_truth=True))
                start_id = graph.state_to_id(env.obs())
                if n_actions == 0:
                    chain_sat = True
                    # apply first skill
                    applied_skill = skill_chain.skills[0]
                    applied_skill_id = 0
                    obs, rew, done, info = skill_chain.skills[0].apply(env, obs, render=render)

                else:
                    #halve uncertainty
                    obs = env.obs(force_update=True)
                    chain_sat = skill_chain.precondition_satisfied(obs)
                    if not chain_sat:
                        fail_states.append(env.obs(ground_truth=True))

                    if chain_sat:
                        logger.debug("  Nominal skill sat")
                        obs, rew, done, info = skill_chain.apply_skill(env,
                                                                obs,
                                                                render=render)
                        applied_skill = info['applied_skill']
                        applied_skill_id = info['applied_skill_id']

                    elif recover:
                        logger.debug("  Executing recovery skill")
                        # Failure
                        # Apply recovery skill
                        # Select recovery skill using the classifier
                        failure_id = failure_clf.predict(obs)
                        state_id = planner.get_state_id(failure_id, cluster=True)
                        logger.debug(f"    Failure id: {failure_id}")

                        # check if recovery makes sense
                        value = V[state_id]
                        failure_value = V[graph.absorb_fail_id]
                        if value > failure_value:
                            action = policy[state_id]
                            if action == 0:
                                # Nominal
                                __import__('ipdb').set_trace()
                            # XXX action = 0 => nominal
                            recovery_id = action - 1
                            all_info['recovery']['hist'].append([failure_id,
                                                                recovery_id])
                            all_info['recovery']['eval_hist'][i].append([failure_id,
                                                                recovery_id])
                            best_recovery_skill = recoveries[failure_id][recovery_id]['skill']
                            best_succ_prob = recoveries[failure_id][recovery_id]['accuracy']
                            logger.debug(f"    Skill to subgoal {recovery_id} with success prob: {best_succ_prob}")

                            all_info['recovery']['times_triggered'] += 1
                            obs, rew, done, info = best_recovery_skill.apply(env,
                                                                        obs,
                                                                        render=render)
                            # gt_obs = env.obs(ground_truth=True)
                            obs = env.obs()
                            sat = preconds[recovery_id].is_satisfied(obs)
                            if sat:
                                all_info['recovery']['success'].append(True)
                            else:
                                all_info['recovery']['success'].append(False)
                                if early_term:
                                    # no cascading failures
                                    T_nominal[start_id, graph.absorb_fail_id] += 1
                                    break
                        else:
                            logger.debug("  Failure state. Breaking")
                            # reward += fail_rew
                            all_info['recovery']['hist'].append([failure_id,
                                                                -1])
                            all_info['recovery']['success'].append(False)
                            T_nominal[start_id, graph.absorb_fail_id] += 1
                            break
                    else:
                        break
                # term_id = graph.state_to_id(env.obs(ground_truth=True))
                term_id = graph.state_to_id(env.obs())
                T_nominal[start_id, term_id] += 1

                n_actions += 1

    transition_p = T_nominal / np.sum(T_nominal, axis=1, keepdims=True)
    all_info['transition_p'] = transition_p

    ood_fails = []
    if failure_clf:
        classified_fail_states = [[] for _ in range(7)]
        for state in fail_states:
            fail_id = failure_clf.predict(state)
            if fail_id == -1:
                logger.info("Out of Distribution Failure")
                ood_fails.append(state)
                continue
            classified_fail_states[fail_id].append(state)
    else:
        classified_fail_states = None
    logger.info(f"Found {len(fail_fails)} failures")
    logger.info(f"Found {len(ood_fails)} ood failures")

    return fail_states, classified_fail_states, all_info
