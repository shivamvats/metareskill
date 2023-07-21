"""
Evaluates recovery skills together with the nominal skills by randomly
generating start states and executing the nominal skill chain. If all nominal
skills are unsat, then the best recovery is picked (based on the recovery
classifier) and executed.

State estimation model:
    1. State estimation error (std) halves after every skill execution.
"""

import copy
import logging
from os.path import join, isfile
import time

import hydra
from hydra.utils import to_absolute_path
from icecream import ic
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import pickle as pkl
from stable_baselines3.common.utils import set_random_seed
from tqdm import trange, tqdm

from recovery_skills.envs.franka_door_env import FrankaDoorEnv
from recovery_skills.envs.franka_door_subtask_env import FrankaDoorSubtaskEnv
from recovery_skills.envs import RayActorSubtaskWrapper, RayVecEnvWrapper
from recovery_skills.skills import SkillChain
from recovery_skills.skills.nominal_door_opening_skills import (
    ReachAndGraspHandleSkill,
    RotateHandleSkill,
    PullHandleSkill,
)
from recovery_skills.skills.nearest_neighbor_skill import NearestNeighborSkill
from recovery_skills.skills.utils import *
from recovery_skills.skills.baselines import *
from recovery_skills.utils import *
from recovery_skills.recovery.value_mab_planner import *
from recovery_skills.graph.symbolic_graph import *
from recovery_skills.recovery.failure_discovery import *
from robosuite.wrappers import VisualizationWrapper

logger = logging.getLogger(__name__)
# logger.setLevel("INFO")
logger.setLevel("DEBUG")


def make_env(cfg, env_cfg):
    controller_cfg = OmegaConf.to_container(env_cfg.controller)
    env = VisualizationWrapper(
        FrankaDoorSubtaskEnv(
            goal_constraint=None,
            controller_configs=controller_cfg,
            has_renderer=cfg.render,
            horizon=env_cfg.horizon,
            context_cfg=env_cfg.context,
            obs_uncertainty=env_cfg.obs_uncertainty,
            timestep=env_cfg.timestep,
            eef_start_region_cfg=env_cfg.eef_start_region,
        ),
        "default",
    )

    return env


def evaluate_recovery_skills_individually(env, nominal_skills, preconds, recovery_policies,
                             failure_clf, cfg):
    # 1. Improvement in overall task success
    # 2. % of times recovery succeeded in activating a nominal skill
    all_info = {'success': 0,
            'failure': 0,
            'hist': []}

    for t in range(cfg.nevals):
        obs = env.reset()

        for i, skill in enumerate(nominal_skills):
            gt_obs = env.obs(ground_truth=True)
            precond_satisfied = skill.precondition_satisfied(gt_obs)
            if i == 0:
                precond_satisfied = True

            if precond_satisfied:
                # run skill chain open loop
                obs, rew, done, info = skill.apply(
                    env, obs, None, render=cfg.render)

            else:
                # failure
                # state update
                gt_obs = env.obs(ground_truth=True)
                recovery_id = failure_clf.predict(gt_obs)
                logger.info(f"    Recovery id: {recovery_id}")
                # recovery ends up at the start of skill_Id
                skill, target_subgoal = recovery_policies[recovery_id]
                logger.info(f"   Executing recovery with subgoal: {target_subgoal}")
                if skill is None:
                    logger.info("  Reached a dead-end!")
                    all_info['failure'] += 1
                    all_info['hist'].append((recovery_id, False))
                    break
                # recovery
                obs, rew, done, info = skill.apply(
                    env, gt_obs, None, render=cfg.render)
                gt_obs = env.obs(ground_truth=True)
                if preconds[target_subgoal].is_satisfied(gt_obs):
                    logger.info("  Recovery successful")
                    all_info['success'] += 1
                    all_info['hist'].append((recovery_id, True))
                else:
                    logger.info("  Recovery failed")
                    all_info['failure'] += 1
                    all_info['hist'].append((recovery_id, False))

    total_failures = all_info['failure'] + all_info['success']
    recovery_rate = all_info['success'] / total_failures * 100
    logger.info("=================")
    logger.info(f"Total evaluations: {cfg.nevals}")
    logger.info(f"Recovered from {all_info['success']} / {total_failures}")
    logger.info(f"Recovery rate: {recovery_rate}%")
    logger.info("=================")

    return all_info


def plot_info(info):
    failure_rate = info['n_fails'] / info['n_evals'] * 100
    recovery_success_rate = np.mean(info['recovery']['success']) * 100
    rews = np.array(info['rewards'])
    exec_costs = np.array(info['exec_costs'])
    success = np.array(info['success'])
    rew_on_success = rews[success]
    exec_costs_on_success = exec_costs[success]

    logger.info("----------Stats---------")
    logger.info(f"  #evals: {info['n_evals']}")
    logger.info(f"  #fails: {info['n_fails']}")
    logger.info(f"  Failure rate: {failure_rate}%")
    logger.info(f"  Reward: {np.mean(rews)} +- {np.std(rews):.3f}")
    logger.info(f"  Reward on success: {np.mean(rew_on_success)} +- {np.std(rew_on_success):.3f}")
    logger.info(f"  Exec cost: {np.mean(exec_costs)} +- {np.std(exec_costs):.3f}")
    logger.info(f"  Exec cost on success: {np.mean(exec_costs_on_success)} +- {np.std(exec_costs_on_success):.3f}")
    logger.info(f"  #recovery triggered: {info['recovery']['times_triggered']}")
    logger.info(f"  #recovery success rate: {recovery_success_rate}%")

    stats = {'failure rate': failure_rate}
    fig, ax = plt.subplots()
    ax.bar(list(stats.keys()),
           list(stats.values()))
    fig.savefig('stats.png')
    plt.close(fig)

    # plot recovery counts
    nclusters = 6
    nsubgoals = 4
    call_hist = np.array(info['recovery']['hist'])
    succ_hist =  np.array(info['recovery']['success'])
    if len(call_hist):
        x, y, texts = [], [], []
        for i in range(nclusters):
            failure_hist = call_hist[call_hist[:,0] == i][:,1]
            succ = succ_hist[call_hist[:,0] == i]
            n_succ = np.sum(succ)
            n_execs = len(succ)
            subgoal_ids, counts = np.unique(failure_hist, return_counts=True)
            if len(failure_hist) > 0:
                x.append(f"{i} to {subgoal_ids[0]}")
                y.append(counts[0])
            else:
                x.append(f"{i} to -")
                y.append(0)
            text = f"{n_succ}/{n_execs} = {n_succ/n_execs:.2f}"
            texts.append(text)

        print("x:, ", x)
        print("y: ", y)
        # __import__('ipdb').set_trace()
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_xlabel("Failure clusters")
        ax.set_ylabel("# executions")
        for i, text in enumerate(texts):
            ax.annotate(text, (i, y[i] + 0.1))
        # TODO also plot recovery success rates
        fig.savefig("recovery_calls.png")
        plt.close(fig)


@hydra.main(config_path="../cfg", config_name="evaluate_recovery_skills")
def main(cfg):
    if cfg.seed is None:
        seed = int(time.time())
    else:
        seed = cfg.seed
    if cfg.task_set is not None:
        seeds = [385, 52, 894, 99, 28, 18]
        seed = seeds[cfg.task_set]

    logger.info(f"Using seed: {seed}")
    logger.info(f"Evaluating recovery skills from {cfg.path_to_recovery_skills}")
    logger.info("------------------------------------------------")
    set_random_seed(seed)

    env_cfg = cfg.env
    env = make_env(cfg, env_cfg)
    _cfg = copy.deepcopy(cfg)
    _cfg.render = False
    _env = make_env(_cfg, env_cfg)

    tasks = []
    for _ in range(cfg.nevals):
        env.reset()
        gt_obs = env.obs(ground_truth=True)
        tasks.append(gt_obs)

    skill_chain = load_nominal_skill_chain(cfg)

    nskills = skill_chain.size
    preconds = pkl_load(cfg.path_to_preconds, True)

    failure_clusters = pkl_load(cfg.path_to_failure_clusters, True)
    failure_clf = pkl_load(cfg.path_to_failure_clf, True)
    recovery_skills = pkl_load(cfg.path_to_recovery_skills, True)
    p_low = pkl_load(cfg.path_to_transition_low, True)

    sym_graph = SymbolicGraph(preconds, failure_clf)

    # if cfg.path_to_transitions:
        # p_transition = pkl_load(cfg.path_to_transitions, True)[-1]
    # else:
        # p_transition, T = estimate_transition_matrix(env, sym_graph,
                                                     # skill_chain, recovery_skills,
                                                     # nevals=cfg.n_transition_evals,
                                                     # MAX_ACTIONS=cfg.max_actions, render=False)
        # p_transition = sym_graph._normalize_transition_matrix(p_transition)
        # logger.debug("  Transition matrix:")
        # logger.debug(f"{p_transition}")
        # pkl_dump((p_transition, T), "p_transition.pkl")

    nclusters = len(recovery_skills)
    nsubgoals = len(preconds)
    MAX_ACTIONS = cfg.max_actions
    MAX_FAILS = cfg.max_fails

    oracle = pkl_load(join(cfg.path_to_results_dir, 'oracle.pkl'), True)
    mab_info = pkl_load(join(cfg.path_to_results_dir, 'mab_info.pkl'), True)
    for i in range(cfg.nrounds):
        # TODO
        pass


    skill_accs = []
    for cluster in recovery_skills:
        accs = []
        for skill in cluster:
            if skill:
                accs.append(skill['accuracy'])
            else:
                accs.append(0.0)
        skill_accs.append(np.array(accs))
    skill_accs = np.array(skill_accs)
    # skill_accs = np.zeros((nclusters, nsubgoals))

    # train_tasks = pkl_load(join(
        # cfg.path_to_tasks_root, f"train_tasks_0.pkl"), True)
    # cluster_sizes = np.array([len(cluster[0]) for cluster in train_tasks])
    # ntotal_failures = sum(cluster_sizes)
    # cluster_ids = np.arange(nclusters)
    # cluster_weights = cluster_sizes / ntotal_failures

    planner = MonoValueMABPlanner(sym_graph, cluster_weights,
                                p_low)
    planner.update_transition_matrix(skill_accs)

    R = sym_graph.construct_reward_matrix()
    policy, V = sym_graph.solve(p_transition, R)

    logger.info(f"Value Function: {V}")
    logger.info(f"Policy: {policy}")
    fail_mode_ids = sym_graph.fail_mode_ids
    hl_policy = np.array(policy[fail_mode_ids]) - 1
    hl_policy_succ = [skill_accs[i, hl_policy[i]] for i in range(nclusters)]
    logger.info(f"High level Policy: {hl_policy}")
    logger.info(f"High level Policy Success: {np.round(hl_policy_succ, 2)}")

    return

    all_info = {'n_evals': 0,
                'n_fails': 0,
                'n_actions': [],
                'success': [],
                'rewards': [],
                'exec_costs': [],
                'mean_reward': 0,
                'skill_accs': skill_accs,
                'hl_policy': hl_policy,
                'hl_policy_succ': hl_policy_succ,
                'recovery': {
                    'times_triggered': 0,
                    'success': [],
                    'hist':[],
                    'eval_hist': [[] for _ in range(cfg.nevals)],
                }
                }
    pkl_dump(all_info, 'info.pkl')

    T_nominal = np.zeros((2, nsubgoals, nsubgoals))

    for i in range(cfg.nevals):
        logger.info(f"Evaluation {i}")
        logger.info("============")
        env.set_obs_corruption(True)
        # obs = env.reset()
        env.reset_from_state(tasks[i])
        obs = env.obs()
        # reward = 0
        all_info['n_evals'] += 1
        all_info['T_nominal'] = T_nominal
        n_actions = 0
        episode_rew = 0
        episode_sym_rew = 0
        recovery_mode = False

        eef_states = [obs['robot_eef:pose/position']]
        applied_skill = None

        if cfg.open_loop:
            for skill in skill_chain.skills:
                obs, rew, done, info = skill.apply(env, obs, render=cfg.render)
                eef_states.append(obs['robot_eef:pose/position'])
                sym_rew = R[sym_graph.state_to_id(env.obs(ground_truth=True))]
                episode_rew += sym_rew
                n_actions += 1
            if not env.check_task_success():
                episode_rew += R[sym_graph.absorb_fail_id]

        else:
            while True:
                logger.info(f"Attempt {n_actions}")
                if n_actions >= MAX_ACTIONS or env.check_task_success():
                    if not env.check_task_success():
                        logger.info("Max actions exceeded\n")
                        episode_rew += R[sym_graph.absorb_fail_id]
                    break

                if n_actions == 0:
                    chain_sat = True
                    # apply first skill
                    applied_skill = skill_chain.skills[0]
                    applied_skill_id = 0
                    obs, rew, done, info = skill_chain.skills[0].apply(env, obs, render=cfg.render)
                    sym_rew = R[sym_graph.state_to_id(env.obs(ground_truth=True))]
                    episode_rew += sym_rew
                    if preconds[applied_skill_id + 1].is_satisfied(obs):
                        T_nominal[1, applied_skill_id, applied_skill_id + 1] += 1
                    else:
                        T_nominal[0, applied_skill_id, applied_skill_id + 1] += 1

                    eef_states.append(obs['robot_eef:pose/position'])

                else:
                    #halve uncertainty
                    obs = env.obs(force_update=True)
                    chain_sat = skill_chain.precondition_satisfied(obs)

                    if chain_sat:
                        logger.info("  Nominal skill sat")
                        recovery_mode = False
                        obs, rew, done, info = skill_chain.apply_skill(env,
                                                                obs,
                                                                render=cfg.render)
                        sym_rew = R[sym_graph.state_to_id(env.obs(ground_truth=True))]
                        episode_rew += sym_rew
                        applied_skill = info['applied_skill']
                        applied_skill_id = info['applied_skill_id']
                        if preconds[applied_skill_id + 1].is_satisfied(obs):
                            T_nominal[1, applied_skill_id, applied_skill_id + 1] += 1
                        else:
                            T_nominal[0, applied_skill_id, applied_skill_id + 1] += 1
                        eef_states.append(obs['robot_eef:pose/position'])

                    elif cfg.recover:

                        logger.info("  Executing recovery skill")
                        recovery_mode = True
                        if cfg.recovery_strategy == 'learnt':
                            # if recovery_mode:
                                # all_info['recovery']['success'].append(False)
                            # Failure
                            # Apply recovery skill
                            # Select recovery skill using the classifier
                            failure_id = failure_clf.predict(obs)
                            state_id = planner.get_state_id(failure_id, cluster=True)
                            logger.info(f"    Failure id: {failure_id}")

                            # check if recovery makes sense
                            value = V[state_id]
                            action = policy[state_id]
                            failure_value = V[sym_graph.absorb_fail_id]

                            if (value > failure_value) and action > 0:
                                action = policy[state_id]
                                # if action == 0:
                                    # action = 0 => nominal
                                    # raise ValueError("Should not have chosen this action")
                                recovery_id = action - 1
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
                                                                            render=cfg.render)

                                sym_rew = R[sym_graph.state_to_id(env.obs(ground_truth=True))]
                                episode_rew += sym_rew
                                eef_states.append(obs['robot_eef:pose/position'])
                                # reward += action_rew
                                gt_obs = env.obs(ground_truth=True)
                                sat = preconds[recovery_id].is_satisfied(gt_obs)
                                if sat:
                                    all_info['recovery']['success'].append(True)
                                else:
                                    all_info['recovery']['success'].append(False)
                            else:
                                logger.info("  Failure state. Breaking")
                                # reward += fail_rew
                                all_info['recovery']['hist'].append([failure_id,
                                                                    -1])
                                all_info['recovery']['success'].append(False)
                                sym_rew = R[sym_graph.absorb_fail_id]
                                episode_rew += sym_rew
                                break

                        elif cfg.recovery_strategy == 'retry':
                            failed_skill = applied_skill
                            recovery_skill = RetrySkill(failed_skill)
                            all_info['recovery']['times_triggered'] += 1
                            obs, rew, done, info = recovery_skill.apply(env,
                                                                        obs,
                                                                        render=cfg.render)
                            sym_rew = R[sym_graph.state_to_id(env.obs(ground_truth=True))]
                            episode_rew += sym_rew
                            eef_states.append(obs['robot_eef:pose/position'])
                            gt_obs = env.obs(ground_truth=True)
                            # sat = applied_skill [recovery_id].is_satisfied(gt_obs)
                            sat = skill_chain.precondition_satisfied(gt_obs)
                            if sat:
                                all_info['recovery']['success'].append(True)
                            else:
                                all_info['recovery']['success'].append(False)

                        elif cfg.recovery_strategy == 'go_to_prev':
                            failed_skill = applied_skill
                            recovery_skill = GoToPrevStateSkill(failed_skill, _env)
                            all_info['recovery']['times_triggered'] += 1
                            obs, rew, done, info = recovery_skill.apply(env,
                                                                        obs,
                                                                        render=cfg.render)
                            sym_rew = R[sym_graph.state_to_id(env.obs(ground_truth=True))]
                            episode_rew += sym_rew
                            eef_states.append(obs['robot_eef:pose/position'])
                            # reward += action_rew
                            gt_obs = env.obs(ground_truth=True)
                            # sat = applied_skill [recovery_id].is_satisfied(gt_obs)
                            sat = skill_chain.precondition_satisfied(gt_obs)
                            if sat:
                                all_info['recovery']['success'].append(True)
                            else:
                                all_info['recovery']['success'].append(False)

                        elif cfg.recovery_strategy == 'go_to_start':
                            target_precond = preconds[0]
                            target_state = target_precond.sample(cls=FrankaDoorEnv,
                                                                n=1)[0]
                            target_rl_state = _env.state_to_rl_state(target_state)
                            # target_eef_pos = target_rl_state['robot_eef:pose/position']
                            target_eef_pos = env_cfg.eef_home.position
                            target_eef_quat = target_rl_state['robot_eef:pose/quat']


                            recovery_skill = GoToStartSkill(target_eef_pos,
                                                            target_eef_quat)
                            all_info['recovery']['times_triggered'] += 1
                            obs, rew, done, info = recovery_skill.apply(env,
                                                                        obs,
                                                                        render=cfg.render)
                            # episode_rew += rew
                            eef_states.append(obs['robot_eef:pose/position'])
                            gt_obs = env.obs(ground_truth=True)
                            sym_rew = R[sym_graph.state_to_id(env.obs(ground_truth=True))]
                            episode_rew += sym_rew
                            sat = skill_chain.precondition_satisfied(gt_obs)
                            if sat:
                                all_info['recovery']['success'].append(True)
                            else:
                                all_info['recovery']['success'].append(False)

                    else:
                        sym_rew = R[sym_graph.absorb_fail_id()]
                        episode_rew += sym_rew
                        break

                n_actions += 1

        all_info['n_actions'].append(n_actions)
        action_cost = 0
        for i in range(1, len(eef_states)):
            cost = np.linalg.norm(eef_states[i] - eef_states[i-1])
            action_cost += cost

        all_info['rewards'].append(episode_rew)
        all_info['exec_costs'].append(action_cost)

        if env.check_task_success():
            logger.info("Success!\n")
            all_info['success'].append(True)
        else:
            all_info['n_fails'] += 1
            all_info['success'].append(False)

        plot_info(all_info)
        pkl_dump(all_info, 'info.pkl')


if __name__ == "__main__":
    main()
