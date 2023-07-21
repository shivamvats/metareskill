from copy import deepcopy, copy
from itertools import cycle
import logging
import math
import time
from os.path import join

import hydra
from hydra.utils import to_absolute_path
from icecream import ic
from omegaconf import OmegaConf
import pickle as pkl
import matplotlib.pyplot as plt
import ray
from sklearn.model_selection import *
from stable_baselines3.common.utils import set_random_seed

from klampt import vis
from robosuite.wrappers import VisualizationWrapper

from recovery_skills.skills import (
    IdentityEESpaceSkill,
    REPSSkill,
)
from recovery_skills.utils import get_rand_in_range, load_pillar_states
from recovery_skills.utils.learning_from_demo import record_demo
from recovery_skills.envs.utils import *
from recovery_skills.graph import State
from recovery_skills.graph.abstraction import *
from recovery_skills.envs.improved_gym_wrapper import ImprovedGymWrapper
from recovery_skills.envs.franka_door_env import FrankaDoorEnv
from recovery_skills.envs.franka_door_subtask_env import FrankaDoorSubtaskEnv
from recovery_skills.envs import RayActorWrapper, RayActorSubtaskWrapper, RayVecEnvWrapper
from recovery_skills.skills.nearest_neighbor_skill import *
from recovery_skills.recovery.failure_discovery import *
from recovery_skills.recovery import *
from recovery_skills.skills.utils import *
from recovery_skills.utils.learning_from_demo import segment_demo
from recovery_skills.utils import *
from recovery_skills.utils.data_processing import *
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def make_env(cfg, env_cfg):
    controller_cfg = OmegaConf.to_container(env_cfg.controller)
    env = FrankaDoorSubtaskEnv(
        goal_constraint=None,
        controller_configs=controller_cfg,
        has_renderer=cfg.render,
        # has_renderer=True,
        horizon=env_cfg.horizon,
        context_cfg=env_cfg.context,
        obs_uncertainty=env_cfg.obs_uncertainty,
        timestep=env_cfg.timestep,
        eef_start_region_cfg=env_cfg.eef_start_region,
    )
    env.set_obs_corruption(False)

    return env

def make_vec_env(cfg, env_cfg):
    controller_cfg = OmegaConf.to_container(env_cfg.controller)
    train_envs =  [
        RayActorSubtaskWrapper.remote(
            goal_constraint=None,
            controller_configs=controller_cfg,
            has_renderer=cfg.render,
            horizon=env_cfg.horizon,
            context_cfg=env_cfg.context,
            obs_uncertainty=env_cfg.obs_uncertainty,
            timestep=env_cfg.timestep,
            eef_start_region_cfg=env_cfg.eef_start_region,
        )
        for _ in range(cfg.train.num_cpus)
    ]
    train_env = RayVecEnvWrapper(train_envs)
    return train_env


def visualize_rl_goal(goal, controller_cfg):
    env = FrankaDoorEnv(controller_cfg, has_renderer=True)
    env = VisualizationWrapper(env, "default")
    env.reset()
    env.set_indicator_pos(
        "indicator0", goal.as_ordered_dict()["robot_eef:pose/position"]
    )
    env.sim.forward()
    for _ in range(500):
        env.render()


def solve_failures_sequentially(failures, nominal_skill_chain, subgoals, cfg):
    """
    Learn recovery skills for failures using REPS
    """

    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    eval_env = make_env(cfg, cfg.env)
    full_skill_chain = SkillChain(
        [ReachAndGraspHandleSkill(), RotateHandleSkill(), PullHandleSkill()]
    )
    viz = False

    all_tasks = failures_to_tasks(failures, nominal_skill_chain)

    skill_id = 0
    subgoal_id = skill_id + 1
    tasks = np.array(all_tasks[subgoal_id], dtype=object)

    recovery_skills = [[] for _ in range(len(failures))]
    failed_recovery_skills = [[] for _ in range(len(failures))]

    # np.random.shuffle(tasks)
    # tasks = tasks.flatten()

    for task_id, task in zip(range(cfg.ntasks), tasks[:, 1]):
        # for task_id, task in zip(range(cfg.ntasks), tasks):
        logger.info(f"Task [{task_id}]:")
        logger.info(f"  Failed skill: {skill_id}, Target subgoal: {task.goal_id}")
        skill = REPSSkill(task.goal_constraint, cfg.env)
        skill.goal_id = task.goal_id
        converged = skill.train_policy(
            task.start["state"],
            controller_cfg=controller_cfg,
            train_cfg=cfg.train,
            reps_cfg=cfg.algo,
            render=cfg.render,
        )

        # check success by executing the rest of the chain
        obs = eval_env.reset_from_state(task.start["state"])
        obs, rew, done, info = skill.apply(
            eval_env, obs, None, deterministic=True, render=viz
        )

        suffix_skill_chain = SkillChain(full_skill_chain.skills[task.goal_id :])

        obs, rew, done, info = suffix_skill_chain.apply(
            eval_env, obs, None, precond_check=False, render=viz
        )
        success = info["is_solved"]
        # success = check_handle_grasped(eval_env)

        if success:
            recovery_skills[subgoal_id].append(skill)
            logger.info("  Successfully learnt recovery skill.")
        else:
            failed_recovery_skills[subgoal_id].append(skill)
            logger.info("  Failed to learn recovery skill.")

        pkl_dump(recovery_skills, "recovery_skills.pkl")
        pkl_dump(failed_recovery_skills, "failed_recovery_skills.pkl")

    # env.set_goal_constraint(goal_constrainst)

    return recovery_skills


def dummy_solve_task(task, cluster_id, arm_id):
    ps = [[0.8, 0.6], [0.8, 0.6], [0.8, 0.6]]
    p = ps[cluster_id][arm_id]
    s = np.random.binomial(1, p)
    return s


def compute_val_accuracy(knn_skill, env, all_tasks):
    ntasks = len(all_tasks)
    all_tasks = copy.copy(all_tasks)
    solved = []
    nenvs = 1
    is_vec_env = hasattr(env, 'envs')
    if is_vec_env:
        nenvs = env.num_envs

    goal = all_tasks[0].goal_constraint
    env.set_goal_constraint(goal)

    # pad tasks
    rem = (len(all_tasks) % nenvs)
    if rem > 0:
        all_tasks.extend(all_tasks[:rem])

    for i in range(len(all_tasks) // nenvs):
        tasks = all_tasks[i*nenvs: (i+1)*nenvs]
        starts = [task.start for task in tasks]
        if is_vec_env:
            obs = env.reset_from_state(starts)
        else:
            obs = env.reset_from_state(starts[0])

        obs, rew, done, info = knn_skill.apply(env, obs, None) #, render=True)
        if is_vec_env:
            solved.extend([info['is_solved'] for info in info])
        else:
            solved.append(info['is_solved'])

    solved = solved[:ntasks]

    return np.mean(solved)


def plot_val_accs(accs, blackbox_info):
    nclusters = len(accs)
    nsubgoals = len(accs[0])

    for cluster_id in range(nclusters):
        fig, ax = plt.subplots()
        cluster_accs = accs[cluster_id]
        call_accs = [datum['call_success_rate'] for datum in blackbox_info[cluster_id]]
        pos_calls= [datum['pos_calls'] for datum in blackbox_info[cluster_id]]
        neg_calls = [datum['neg_calls'] for datum in blackbox_info[cluster_id]]
        X = np.arange(nsubgoals)
        ax.bar(X, cluster_accs, alpha=0.5, label='Skill Accuracy')
        ax.scatter(X, call_accs, color='k', marker='x', label='Blackbox Accuracy')
        for i, (pos, neg, acc) in enumerate(zip(pos_calls, neg_calls, call_accs)):
            text = f"{pos}/{pos + neg}"
            ax.annotate(text, (i, acc + 0.1))

        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.savefig(f"Validation-accuracy_{cluster_id}.png")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(accs, vmin=0, vmax=1.0)
    for (j, i), acc in np.ndenumerate(accs):
        ax.text(i, j, np.round(acc, 2), ha='center', va='center')
    ax.set_title("Recovery accuracy")
    ax.set_xlabel("Subgoals")
    ax.set_ylabel("Failure Clusters")
    fig.savefig("val_accs.png")
    plt.close(fig)



def compute_values(ps, subgoal_v):
    action_reward = -1
    fail_v = subgoal_v[-1]
    values = []
    for p, v in zip(ps, subgoal_v):
        value = action_reward + p*v + (1 - p)*fail_v
        values.append(value)
    logger.info(f"  Failure values: {values}")
    return values


def solve_failures_iteratively(recovery_tasks_train, recovery_tasks_val,
                                      nominal_skill_chain,
                                      subgoals, cfg,
                               ):
    """
    Learn recovery skills for failures using REPS
    """

    from recovery_skills.recovery.value_mab_planner import MonoValueMABPlanner
    from recovery_skills.recovery.round_robin_planner import RoundRobinPlanner
    from recovery_skills.recovery.transition_ucb_planner import TransitionUCBPlanner

    if cfg.planner == "Iterative-Round-Robin":
        round_robin = True
    else:
        round_robin = False

    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    eval_env = make_env(cfg, cfg.env)
    env_cfg = deepcopy(cfg.env)
    env_cfg["obs_uncertainty"]["handle:pose/position"]["std"] = 0.02
    discovery_env = make_env(cfg, env_cfg)
    p_transition = pkl_load(cfg.path_to_transition_low, True)
    failure_clf = pkl_load(cfg.path_to_failure_clf, True)

    nclusters = len(recovery_tasks_train)
    nsubgoals = len(subgoals)
    budget = cfg.budget.nmax
    # round_budget = int(np.ceil(budget / cfg.iterative.n_rounds))

    # plotting
    mab_rew_hist = []
    mab_ucb_hist = []

    dones = np.zeros((nclusters, nsubgoals), dtype=bool)

    recovery_skills = [[[] for _ in range(nsubgoals)] for _ in range(nclusters)]
    failed_recovery_skills = [
        [[] for _ in range(nsubgoals)] for _ in range(nclusters)
    ]
    knn_skills = [[{} for _ in range(nsubgoals)] for _ in range(nclusters)]
    val_accs = np.array([[0.0 for _ in range(nsubgoals)] for _ in range(nclusters)])
    val_accs_hist = [[[] for _ in range(nsubgoals)] for _ in range(nclusters)]
    nsamples = [[0 for _ in range(nsubgoals)] for _ in range(nclusters)]
    failure_value_hist = []
    p_transitions = []
    oracle_hist = [[[] for _ in range(nsubgoals)] for _ in range(nclusters)]
    val_freq = 1
    all_info = {
        'hist': {
            'val_accs': []
        }
    }

    mab_test = False

    if cfg.test:
        # o_tasks_train_1 = pkl_load(join(cfg.path_to_oracle, "tasks_train_1.pkl"))
        # o_tasks_val_1 = pkl_load(join(cfg.path_to_oracle, "tasks_val_1.pkl"))
        # o_p_transition_1 = pkl_load(join(cfg.path_to_oracle, "p_transitions.pkl"))[-1]
        # o_oracle_hist = pkl_load(join(cfg.path_to_oracle, "oracle_hist.pkl"))
        o_recovery_skills = pkl_load(join(cfg.path_to_oracle, "recovery_skills.pkl"), True)
        o_failed_recovery_skills = pkl_load(join(cfg.path_to_oracle, "failed_recovery_skills.pkl"), True)

    logger.info("==========================")
    logger.info(f"Solving for {nclusters} failure clusters with {nsubgoals} potential recoveries")
    logger.info(f"  Total potential recoveries: {nclusters*nsubgoals}")
    logger.info(f"  Total Budget: {budget}")
    logger.info(f"  Rounds: {cfg.iterative.n_rounds}")
    logger.info(f"  Estimated transition mat: {p_transition}")

    logger.info(f"  Iterative")
    logger.info("==========================")

    cluster_weights = None
    cluster_ids = np.arange(nclusters)
    all_tasks_train, all_tasks_val = [], []

    for round_idx in range(cfg.iterative.n_rounds):
        logger.info(f"Round {round_idx}")
        logger.info("=============")

        sym_graph = SymbolicGraph(subgoals, failure_clf)
        discovery_planner = MonoValueMABPlanner(sym_graph,
                                                cluster_weights,
                                                p_transition,
                                                c=cfg.MAB.c,
                                                initialize=cfg.MAB.initialize,
                                                discount=cfg.MAB.discount)

        if round_idx == 0:
            _, failures, info = discover_failures(discovery_env,
                                               nominal_skill_chain,
                                               None,
                                               discovery_planner,
                                               subgoals,
                                               failure_clf,
                                               cfg.iterative.n_fails_per_round,
                                               # 10,
                                               # pessimistic_discovery=True,
                                               pessimistic_discovery=False,
                                               early_term=True)
            p_transition = info['transition_p']
            round_budget = cfg.iterative.round_0_budget
            logger.warning("  Using saved failures")
            tasks_train, tasks_val = recovery_tasks_train, recovery_tasks_val
            for i in range(nclusters):
                all_tasks_train.append([list(tasks) for tasks in tasks_train[i]])
                all_tasks_val.append([list(tasks) for tasks in tasks_val[i]])

            if cfg.planner == "Iterative-Round-Robin":
                planner = RoundRobinPlanner(nclusters, nsubgoals, budget)
            elif cfg.planner == "Iterative-Mono-MAB":
                raise NotImplementedError
                    # planner = MonoValueMABPlanner(sym_graph,
                                                # cluster_weights,
                                                # p_transition,
                                                # c=cfg.MAB.c,
                                                # initialize=cfg.MAB.initialize,
                                                # discount=cfg.MAB.discount)
            elif cfg.planner == "Transition-UCB":
                planner = TransitionUCBPlanner(sym_graph,
                                            p_transition,
                                            c=cfg.MAB.c,
                                            initialize=cfg.MAB.initialize,
                                            planning_horizon=cfg.planning_horizon,
                                            discount=cfg.MAB.discount)
            else:
                raise NotImplementedError
            pkl_dump(planner, "planner.pkl")

        else:
            # if not cfg.test:
            _, failures, info = discover_failures(discovery_env,
                                            nominal_skill_chain,
                                            knn_skills,
                                            discovery_planner,
                                            subgoals, failure_clf,
                                            cfg.iterative.n_fails_per_round,
                                            pessimistic_discovery=False)
            p_transition = info['transition_p']
            round_budget = budget - cfg.iterative.round_0_budget

            all_tasks = failures_to_tasks(failures, nominal_skill_chain, subgoals)

            tasks_train, tasks_val = split_into_train_and_val_tasks(all_tasks,
                                                                    cluster_ids,
                                                                    nclusters,
                                                                    nsubgoals)
            # else:
                # tasks_train, tasks_val = o_tasks_train_1, o_tasks_val_1
                # p_transition = o_p_transition_1

            for i in range(nclusters):
                for j in range(nsubgoals):
                    all_tasks_train[i][j].extend(tasks_train[i][j])
                    all_tasks_val[i][j].extend(tasks_val[i][j])

            if cfg.planner == 'Transition-UCB':
                planner.update_transition_matrix(p_transition)

        p_transitions.append(p_transition)

        pkl_dump(p_transitions, "p_transitions.pkl")
        pkl_dump(tasks_train, f"tasks_train_{round_idx}.pkl")
        pkl_dump(tasks_val, f"tasks_val_{round_idx}.pkl")

        cluster_sizes = np.array([len(cluster[0]) for cluster in tasks_train])
        ntotal_failures = sum(cluster_sizes)
        cluster_weights = cluster_sizes / ntotal_failures
        logger.info(f"  Cluster sizes: {cluster_sizes}")

        for t in range(round_budget):
            cluster_id, subgoal_id = planner.pull()
            logger.info(f"  [{t}] Next recovery: cluster {cluster_id}, subgoal {subgoal_id}")
            logger.info("-------------------------")

            cluster = all_tasks_train[cluster_id]

            iter = 0
            while iter < cfg.budget.oracle_calls_per_iter:
                skill = recovery_skills[cluster_id][subgoal_id]

                if len(cluster[subgoal_id]) == 0:
                    dones[cluster_id][subgoal_id] = True
                    break

                task = cluster[subgoal_id].pop()

                if mab_test:
                    good_arms = [0, 5, 9]
                    arm_id = planner.get_arm_id(cluster_id, subgoal_id)
                    if arm_id in good_arms:
                        val_accs[cluster_id][subgoal_id] = min(1.0,
                                                            val_accs[cluster_id][subgoal_id]
                                                            + 0.05)
                    iter += 1

                else:
                    if not cfg.test:
                        skill = REPSSkill(task.goal_constraint, cfg.env)
                        subgoal_id = task.goal_id
                        converged, info = skill.train_policy(
                            task.start,
                            controller_cfg=controller_cfg,
                            train_cfg=cfg.train,
                            reps_cfg=cfg.algo,
                            render=cfg.render,
                        )
                        iter += 1
                        solved = converged

                    else:
                        logger.info("  ")
                        # solved, skill = o_oracle_hist[cluster_id][subgoal_id].pop()
                        o_pos_samples = o_recovery_skills[cluster_id][subgoal_id]
                        o_neg_samples = o_failed_recovery_skills[cluster_id][subgoal_id]
                        n_pos, n_neg = len(o_pos_samples), len(o_neg_samples)

                        if n_pos + n_neg <= 0:
                            solved = False
                            skill = None
                        else:
                            o_sample_id = np.random.randint(0, n_pos + n_neg)
                            if o_sample_id < n_pos:
                                solved = True
                                skill = o_pos_samples.pop(0)
                            else:
                                solved = False
                                skill = o_neg_samples.pop(0)
                        iter += 1

                    if solved:
                        recovery_skills[cluster_id][subgoal_id].append(skill)
                        nsamples[cluster_id][subgoal_id] += 1
                        oracle_hist[cluster_id][subgoal_id].append((True, skill))
                        logger.info("  Successfully learnt recovery skill.")

                    else:
                        failed_recovery_skills[cluster_id][subgoal_id].append(skill)
                        oracle_hist[cluster_id][subgoal_id].append((False, skill))
                        logger.info("  Failed to learn recovery skill.")

                    pkl_dump(oracle_hist, "oracle_hist.pkl")
                    pkl_dump(recovery_skills, "recovery_skills.pkl")
                    pkl_dump(failed_recovery_skills, "failed_recovery_skills.pkl")

                if (nsamples[cluster_id][subgoal_id] >= val_freq and
                    len(recovery_skills[cluster_id][subgoal_id]) >= 1):
                    knn_skill = NearestNeighborSkill(
                        recovery_skills[cluster_id][subgoal_id],
                        subgoals[subgoal_id],
                        cfg.env,
                    )
                    val_accuracy = compute_val_accuracy(
                        knn_skill,
                        eval_env,
                        all_tasks_val[cluster_id][subgoal_id],
                    )
                    if val_accuracy >= val_accs[cluster_id][subgoal_id]:
                        val_accs[cluster_id][subgoal_id] = val_accuracy
                        knn_skills[cluster_id][subgoal_id] = {'skill': knn_skill,
                                                                'accuracy': val_accs[cluster_id][subgoal_id]}
                    else:
                        logger.debug("    No improvement in recovery accuracy. Dropping sample")

                    nsamples[cluster_id][subgoal_id] = 0
                    pkl_dump(knn_skills, 'knn_skills.pkl')

                    logger.info(f"  Updated val accuracy to {val_accuracy}")

                    # info
                    def compute_blackbox_info():
                        info = [[{} for _ in range(nsubgoals)] for _ in range(nclusters)]
                        # For each edge
                        for _cluster_id in range(nclusters):
                            for _subgoal_id in range(nsubgoals):
                                _info = info[_cluster_id][_subgoal_id]
                                pos_calls = len(recovery_skills[_cluster_id][_subgoal_id])
                                neg_calls = len(failed_recovery_skills[_cluster_id][_subgoal_id])
                                if pos_calls + neg_calls > 0:
                                    call_success_rate = pos_calls / (pos_calls + neg_calls)
                                else:
                                    call_success_rate = 0
                                _info['pos_calls'] = pos_calls
                                _info['neg_calls'] = neg_calls
                                _info['call_success_rate'] = call_success_rate
                        return info

                    blackbox_info  = compute_blackbox_info()
                    plot_val_accs(val_accs, blackbox_info)
                    pkl_dump(blackbox_info, 'blackbox_info.pkl')

                else:
                    val_accuracy = val_accs[cluster_id][subgoal_id]

                val_accs_hist[cluster_id][subgoal_id].append(val_accuracy)
                pkl_dump(val_accs_hist, 'val_accuracy_hist.pkl')

                # Terminate arm if recovery learnt
                val_thresh = 0.99
                if val_accuracy >= val_thresh:
                    dones[cluster_id][subgoal_id] = True
                    planner.set_arm_done(cluster_id, subgoal_id)
                    logger.debug(f"  Cluster {cluster_id} to subgoal {subgoal_id} solved")
                    logger.debug(f"  Done arms: {np.where(planner.dones)[0]}")

            if np.any(np.isnan(val_accs)):
                import ipdb; ipdb.set_trace();
            planner.update(cluster_id, subgoal_id, val_accs)
            all_info['hist']['val_accs'].append(deepcopy(val_accs))
            pkl_dump(all_info, "info.pkl")
            pkl_dump(planner, "planner.pkl")

            if round_idx != 0 and not round_robin:
                failure_value_hist.append(planner.v_best)
                pkl_dump(failure_value_hist, 'failure_value_hist.pkl')
            if cfg.planner == 'Transition-UCB':
                fig, ax = plt.subplots()
                ax.set_title("Start Value")
                ax.plot(planner.info['hist']['V'])
                plt.savefig("start_value.png")
                plt.close(fig)
            mab_info = planner.info_dict()
            pkl_dump(mab_info, 'mab_info.pkl')
            pkl_dump(recovery_skills, f"recovery_skills_round_{round_idx}.pkl")
            pkl_dump(knn_skills, f"knn_skills_round_{round_idx}.pkl")

            # Plots
            # Plot failure value
            fig, ax = plt.subplots()
            ax.plot(failure_value_hist)
            ax.set_title("Failure value")
            plt.savefig("failure_values.png")
            plt.close(fig)

            ## Plot arm pull history
            narms = planner.narms
            arm_pulls_hist = planner.arm_pulls_hist

            arm_ids = np.arange(narms)
            arm_pulls = np.zeros((narms, len(arm_pulls_hist)))
            for i, pull in enumerate(arm_pulls_hist):
                arm_pulls[arm_pulls_hist, np.arange(len(arm_pulls_hist))] = 1

            # plot arm pulls
            fig, ax = plt.subplots()
            for arm_id in arm_ids:
                ax.plot(
                    np.arange(len(arm_pulls_hist)),
                    np.cumsum(arm_pulls[arm_id]),
                    label=f"State {arm_id}",
                )
                ax.set_xlabel("Episode")
                ax.set_ylabel("# RL calls")
                ax.legend()
            fig.savefig(f"mab_arm_pulls_hist.png")
            plt.close(fig)

            arm_pulls = np.array([planner.arm_pulls(i) for i in
                                range(planner.narms)]).reshape(nclusters,
                                                                nsubgoals)
            fig, ax = plt.subplots()
            ax.imshow(arm_pulls)
            for (j, i), label in np.ndenumerate(arm_pulls):
                ax.text(i, j, label, ha='center', va='center')
            ax.set_title("Arm pulls")
            ax.set_xlabel("Subgoals")
            ax.set_ylabel("Failure Clusters")
            fig.savefig("arm_pulls.png")
            plt.close(fig)

            if not round_robin:
                # plot reward estimates
                mean_rews = planner.rews
                # mean_rews = [np.mean(rew) for rew in rews]
                upper_confidence_bounds = planner.upper_confidence_bounds()

                mab_rew_hist.append(mean_rews)
                mab_ucb_hist.append(upper_confidence_bounds)
                pkl_dump(mab_rew_hist, "mab_rew_hist.pkl")
                pkl_dump(mab_ucb_hist, "mab_ucb_hist.pkl")

                rew_hist = np.array(mab_rew_hist)
                ucb_hist = np.array(mab_ucb_hist)

                fig, ax = plt.subplots()
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward estimate (UCB)")

                arm_ids = np.arange(cluster_id*nsubgoals,
                                    (cluster_id+1)*nsubgoals)
                for j, arm_id in enumerate(arm_ids):
                    means = rew_hist[:, arm_id]
                    ucbs = ucb_hist[:, arm_id]
                    ax.plot(means, label=f"State {j}")
                    ax.fill_between(np.arange(len(means)), means, ucbs, alpha=0.1)
                ax.legend()

                fig.savefig(f"mab_rews_{cluster_id}.png")
                plt.close(fig)

    return recovery_skills


# def filter_failures_and_update_preconds(failures, skill_chain, cfg):
    # env = make_env(cfg, cfg.env)
    # nsubgoals = skill_chain.size + 1
    # init_set_updates = [[] for _ in range(nsubgoals)]
    # rl_init_set_updates = [[] for _ in range(nsubgoals)]
    # y_updates = [[] for _ in range(nsubgoals)]
    # true_failures = [[] for _ in range(nsubgoals)]

    # subgoal_id = 1
    # suffix_skill_chain = SkillChain(skill_chain.skills[subgoal_id:])

    # for failure in tqdm(failures[subgoal_id]):
        # # Execute the chain and verify if it fails
        # obs = env.reset_from_state(failure["state"])
        # obs, rew, done, info = suffix_skill_chain.apply(
            # env, obs, precond_check=False, render=cfg.render
        # )

        # init_set_updates[subgoal_id].append(failure["state"])
        # rl_init_set_updates[subgoal_id].append(failure["rl_state"])
        # y_updates[subgoal_id].append(info["is_solved"])

    # y_updates = np.array(y_updates)

    # for i in range(nsubgoals):
        # true_failures[i] = np.array(failures[i])[np.logical_not(y_updates[i])]

    # for i, skill in enumerate(skill_chain.skills):
        # if len(y_updates[i]) < 5:
            # continue

        # if skill.preconds is None:
            # skill.preconds = PreconditionClassifier(
                # init_set_updates[i],
                # rl_init_set_updates[i],
                # y_updates[i],
            # )

        # else:
            # skill.preconds.update(
                # init_set_updates[i], rl_init_set_updates[i], y_updates[i]
            # )

    # pkl_dump(skill_chain, "updated_skill_chain.pkl")
    # pkl_dump(true_failures, "true_failures.pkl")
    # return true_failures, skill_chain


def solve_failures(recovery_tasks_train, recovery_tasks_val,
                                      nominal_skill_chain,
                                      subgoals, cfg,
                               ):
    """
    Learn recovery skills for failures using REPS
    """

    from recovery_skills.recovery.value_mab_planner import MonoValueMABPlanner
    from recovery_skills.recovery.round_robin_planner import RoundRobinPlanner
    from recovery_skills.recovery.transition_ucb_planner import TransitionUCBPlanner

    if cfg.planner == "Round-Robin":
        round_robin = True
    else:
        round_robin = False

    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    eval_env = make_env(cfg, cfg.env)
    # eval_env = make_vec_env(cfg, cfg.env)
    env_cfg = deepcopy(cfg.env)
    env_cfg["obs_uncertainty"]["handle:pose/position"]["std"] = 0.02
    discovery_env = make_env(cfg, env_cfg)
    # p_transition = pkl_load(cfg.path_to_transition_low, True)
    failure_clf = pkl_load(cfg.path_to_failure_clf, True)

    nclusters = len(recovery_tasks_train)
    nsubgoals = len(subgoals)
    budget = cfg.budget.nmax

    # plotting
    mab_rew_hist = []
    mab_ucb_hist = []

    dones = np.zeros((nclusters, nsubgoals), dtype=bool)

    recovery_skills = [[[] for _ in range(nsubgoals)] for _ in range(nclusters)]
    failed_recovery_skills = [
        [[] for _ in range(nsubgoals)] for _ in range(nclusters)
    ]
    knn_skills = [[[] for _ in range(nsubgoals)] for _ in range(nclusters)]
    val_accs = np.array([[0.0 for _ in range(nsubgoals)] for _ in range(nclusters)])
    val_accs_hist = [[[] for _ in range(nsubgoals)] for _ in range(nclusters)]
    nsamples = [[0 for _ in range(nsubgoals)] for _ in range(nclusters)]
    failure_value_hist = []
    oracle_hist = [[[] for _ in range(nsubgoals)] for _ in range(nclusters)]
    val_freq = 1
    all_info = {
        'hist': {
            'val_accs': []
        }
    }

    mab_test = False

    if cfg.test or cfg.warm_start:
        # o_tasks_train_1 = pkl_load(join(cfg.path_to_oracle, "tasks_train_1.pkl"))
        # o_tasks_val_1 = pkl_load(join(cfg.path_to_oracle, "tasks_val_1.pkl"))
        # o_p_transition_1 = pkl_load(join(cfg.path_to_oracle, "p_transitions.pkl"))[-1]
        o_oracle_hist = pkl_load(join(
            cfg.path_to_oracle, "oracle_hist.pkl"), True)
        # o_recovery_skills = pkl_load(join(cfg.path_to_oracle, "recovery_skills.pkl"), True)
        # o_failed_recovery_skills = pkl_load(join(cfg.path_to_oracle, "failed_recovery_skills.pkl"), True)

    cluster_weights = None
    cluster_ids = np.arange(nclusters)

    sym_graph = SymbolicGraph(subgoals, failure_clf)
    p_transition = construct_optimistic_transition_mat(sym_graph)

    logger.info("==========================")
    logger.info(f"Solving for {nclusters} failure clusters with {nsubgoals} potential recoveries")
    logger.info(f"  Total potential recoveries: {nclusters*nsubgoals}")
    logger.info(f"  Total Budget: {budget}")
    logger.info(f"  Optimistic transition mat: {p_transition}")
    logger.info("==========================")

    all_tasks_train = recovery_tasks_train
    all_tasks_val = recovery_tasks_val

    cluster_sizes = np.array([len(cluster[0]) for cluster in all_tasks_train])
    ntotal_failures = sum(cluster_sizes)
    cluster_weights = cluster_sizes / ntotal_failures

    if cfg.planner == "Round-Robin":
        planner = RoundRobinPlanner(nclusters, nsubgoals, budget)
    elif cfg.planner == "Mono-MAB":
        planner = MonoValueMABPlanner(sym_graph,
                                    cluster_weights,
                                    p_transition,
                                    c=cfg.MAB.c,
                                    initialize=cfg.MAB.initialize,
                                    discount=cfg.MAB.discount)
    elif cfg.planner == "Transition-UCB":
        planner = TransitionUCBPlanner(sym_graph,
                                    p_transition,
                                    cluster_weights,
                                    c=cfg.MAB.c,
                                    initialize=cfg.MAB.initialize,
                                    planning_horizon=cfg.planning_horizon,
                                    discount=cfg.MAB.discount,
                                    window=cfg.window)
    else:
        raise NotImplementedError
    pkl_dump(planner, "planner.pkl")

    for t in range(budget):
        cluster_id, subgoal_id = planner.pull()
        logger.info(f"  [{t}] Next recovery: cluster {cluster_id}, subgoal {subgoal_id}")
        logger.info("-------------------------")

        cluster = all_tasks_train[cluster_id]

        iter = 0
        while iter < cfg.budget.oracle_calls_per_iter:
            skill = recovery_skills[cluster_id][subgoal_id]

            if len(cluster[subgoal_id]) == 0:
                dones[cluster_id][subgoal_id] = True
                break

            task = cluster[subgoal_id].pop()

            if mab_test:
                good_arms = [0, 5, 9]
                arm_id = planner.get_arm_id(cluster_id, subgoal_id)
                if arm_id in good_arms:
                    val_accs[cluster_id][subgoal_id] = min(1.0,
                                                        val_accs[cluster_id][subgoal_id]
                                                        + 0.05)
                iter += 1

            else:
                if cfg.test:
                    solved, skill = o_oracle_hist[cluster_id][subgoal_id].pop(0)
                    iter += 1
                else:
                    if cfg.warm_start and len(o_oracle_hist[cluster_id][subgoal_id]):
                        solved, skill = o_oracle_hist[cluster_id][subgoal_id].pop(0)
                    else:
                        skill = REPSSkill(task.goal_constraint, cfg.env)
                        subgoal_id = task.goal_id
                        converged, info = skill.train_policy(
                            task.start,
                            controller_cfg=controller_cfg,
                            train_cfg=cfg.train,
                            reps_cfg=cfg.algo,
                            render=cfg.render,
                        )
                        solved = converged

                    iter += 1

                if solved:
                    recovery_skills[cluster_id][subgoal_id].append(skill)
                    nsamples[cluster_id][subgoal_id] += 1
                    oracle_hist[cluster_id][subgoal_id].append((True, skill))
                    logger.info("  Successfully learnt recovery skill.")

                else:
                    failed_recovery_skills[cluster_id][subgoal_id].append(skill)
                    oracle_hist[cluster_id][subgoal_id].append((False, skill))
                    logger.info("  Failed to learn recovery skill.")

                pkl_dump(oracle_hist, "oracle_hist.pkl")
                pkl_dump(recovery_skills, "recovery_skills.pkl")
                pkl_dump(failed_recovery_skills, "failed_recovery_skills.pkl")

            if (nsamples[cluster_id][subgoal_id] >= val_freq and
                len(recovery_skills[cluster_id][subgoal_id]) >= 1):
                knn_skill = NearestNeighborSkill(
                    recovery_skills[cluster_id][subgoal_id],
                    subgoals[subgoal_id],
                    cfg.env,
                )

                if round_robin:
                    val_accuracy = val_accs[cluster_id][subgoal_id]
                    knn_skills[cluster_id][subgoal_id] = {'skill': knn_skill,
                                                          'accuracy': None }
                else:
                    # Only Transition-UCB needs to compute val accuracy
                    val_accuracy = compute_val_accuracy(
                        knn_skill,
                        eval_env,
                        all_tasks_val[cluster_id][subgoal_id],
                    )

                    if val_accuracy >= val_accs[cluster_id][subgoal_id]:
                        val_accs[cluster_id][subgoal_id] = val_accuracy
                        knn_skills[cluster_id][subgoal_id] = {
                            'skill': knn_skill,
                            'accuracy': val_accs[cluster_id][subgoal_id]}
                    # else:
                        # logger.debug("    No improvement in recovery accuracy. Dropping sample")

                nsamples[cluster_id][subgoal_id] = 0
                pkl_dump(knn_skills, 'knn_skills.pkl')

                logger.info(f"  Updated val accuracy to {val_accuracy}")

                # info
                def compute_blackbox_info():
                    info = [[{} for _ in range(nsubgoals)] for _ in range(nclusters)]
                    # For each edge
                    for _cluster_id in range(nclusters):
                        for _subgoal_id in range(nsubgoals):
                            _info = info[_cluster_id][_subgoal_id]
                            pos_calls = len(recovery_skills[_cluster_id][_subgoal_id])
                            neg_calls = len(failed_recovery_skills[_cluster_id][_subgoal_id])
                            if pos_calls + neg_calls > 0:
                                call_success_rate = pos_calls / (pos_calls + neg_calls)
                            else:
                                call_success_rate = 0
                            _info['pos_calls'] = pos_calls
                            _info['neg_calls'] = neg_calls
                            _info['call_success_rate'] = call_success_rate
                    return info

                blackbox_info  = compute_blackbox_info()
                plot_val_accs(val_accs, blackbox_info)
                pkl_dump(blackbox_info, 'blackbox_info.pkl')
            else:
                val_accuracy = val_accs[cluster_id][subgoal_id]

            val_accs_hist[cluster_id][subgoal_id].append(val_accuracy)
            pkl_dump(val_accs_hist, 'val_accuracy_hist.pkl')

            # Terminate arm if recovery learnt
            # XXX Disable arm termination
            # val_thresh = 1.99
            # if val_accuracy >= val_thresh:
                # dones[cluster_id][subgoal_id] = True
                # planner.set_arm_done(cluster_id, subgoal_id)
                # logger.debug(f"  Cluster {cluster_id} to subgoal {subgoal_id} solved")
                # logger.debug(f"  Done arms: {np.where(planner.dones)[0]}")

        if np.any(np.isnan(val_accs)):
            import ipdb; ipdb.set_trace();

        planner.update(cluster_id, subgoal_id, val_accs)
        all_info['hist']['val_accs'].append(deepcopy(val_accs))
        pkl_dump(all_info, "info.pkl")
        pkl_dump(planner, "planner.pkl")

        if not round_robin:
            failure_value_hist.append(planner.v_best)
            pkl_dump(failure_value_hist, 'failure_value_hist.pkl')
        if cfg.planner == 'Transition-UCB':
            fig, ax = plt.subplots()
            ax.set_title("Start Value")
            ax.plot(planner.info['hist']['V'])
            plt.savefig("start_value.png")
            plt.close(fig)
        mab_info = planner.info_dict()
        pkl_dump(mab_info, 'mab_info.pkl')
        pkl_dump(recovery_skills, f"recovery_skills.pkl")
        pkl_dump(knn_skills, f"knn_skills.pkl")

        # Plots
        # Plot failure value
        fig, ax = plt.subplots()
        ax.plot(failure_value_hist)
        ax.set_title("Failure value")
        plt.savefig("failure_values.png")
        plt.close(fig)

        ## Plot arm pull history
        narms = planner.narms
        arm_pulls_hist = planner.arm_pulls_hist

        arm_ids = np.arange(narms)
        arm_pulls = np.zeros((narms, len(arm_pulls_hist)))
        for i, pull in enumerate(arm_pulls_hist):
            arm_pulls[arm_pulls_hist, np.arange(len(arm_pulls_hist))] = 1

        # plot arm pulls
        fig, ax = plt.subplots()
        for arm_id in arm_ids:
            ax.plot(
                np.arange(len(arm_pulls_hist)),
                np.cumsum(arm_pulls[arm_id]),
                label=f"State {arm_id}",
            )
            ax.set_xlabel("Episode")
            ax.set_ylabel("# RL calls")
            ax.legend()
        fig.savefig(f"mab_arm_pulls_hist.png")
        plt.close(fig)

        arm_pulls = np.array([planner.arm_pulls(i) for i in
                            range(planner.narms)]).reshape(nclusters,
                                                            nsubgoals)
        fig, ax = plt.subplots()
        ax.imshow(arm_pulls)
        for (j, i), label in np.ndenumerate(arm_pulls):
            ax.text(i, j, label, ha='center', va='center')
        ax.set_title("Arm pulls")
        ax.set_xlabel("Subgoals")
        ax.set_ylabel("Failure Clusters")
        fig.savefig("arm_pulls.png")
        plt.close(fig)

        if not round_robin:
            # plot reward estimates
            mean_rews = planner.rews
            # mean_rews = [np.mean(rew) for rew in rews]
            upper_confidence_bounds = planner.upper_confidence_bounds()

            mab_rew_hist.append(mean_rews)
            mab_ucb_hist.append(upper_confidence_bounds)
            pkl_dump(mab_rew_hist, "mab_rew_hist.pkl")
            pkl_dump(mab_ucb_hist, "mab_ucb_hist.pkl")

            rew_hist = np.array(mab_rew_hist)
            ucb_hist = np.array(mab_ucb_hist)

            fig, ax = plt.subplots()
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward estimate (UCB)")

            arm_ids = np.arange(cluster_id*nsubgoals,
                                (cluster_id+1)*nsubgoals)
            for j, arm_id in enumerate(arm_ids):
                means = rew_hist[:, arm_id]
                ucbs = ucb_hist[:, arm_id]
                ax.plot(means, label=f"State {j}")
                ax.fill_between(np.arange(len(means)), means, ucbs, alpha=0.1)
            ax.legend()

            fig.savefig(f"mab_rews_{cluster_id}.png")
            plt.close(fig)

    return recovery_skills


# def filter_failures_and_update_preconds(failures, skill_chain, cfg):
    # env = make_env(cfg, cfg.env)
    # nsubgoals = skill_chain.size + 1
    # init_set_updates = [[] for _ in range(nsubgoals)]
    # rl_init_set_updates = [[] for _ in range(nsubgoals)]
    # y_updates = [[] for _ in range(nsubgoals)]
    # true_failures = [[] for _ in range(nsubgoals)]

    # subgoal_id = 1
    # suffix_skill_chain = SkillChain(skill_chain.skills[subgoal_id:])

    # for failure in tqdm(failures[subgoal_id]):
        # # Execute the chain and verify if it fails
        # obs = env.reset_from_state(failure["state"])
        # obs, rew, done, info = suffix_skill_chain.apply(
            # env, obs, precond_check=False, render=cfg.render
        # )

        # init_set_updates[subgoal_id].append(failure["state"])
        # rl_init_set_updates[subgoal_id].append(failure["rl_state"])
        # y_updates[subgoal_id].append(info["is_solved"])

    # y_updates = np.array(y_updates)

    # for i in range(nsubgoals):
        # true_failures[i] = np.array(failures[i])[np.logical_not(y_updates[i])]

    # for i, skill in enumerate(skill_chain.skills):
        # if len(y_updates[i]) < 5:
            # continue

        # if skill.preconds is None:
            # skill.preconds = PreconditionClassifier(
                # init_set_updates[i],
                # rl_init_set_updates[i],
                # y_updates[i],
            # )

        # else:
            # skill.preconds.update(
                # init_set_updates[i], rl_init_set_updates[i], y_updates[i]
            # )

    # pkl_dump(skill_chain, "updated_skill_chain.pkl")
    # pkl_dump(true_failures, "true_failures.pkl")
    # return true_failures, skill_chain

@hydra.main(config_path="../cfg", config_name="learn_recovery_skills")
def main(cfg):
    if cfg.seed is None:
        seed = int(time.time())
    else:
        seed = cfg.seed
    if cfg.task_set is not None:
        seeds = [85, 2, 94, 9, 288, 273]
        seed = seeds[cfg.task_set]
    logger.info(f"Using seed: {seed}")
    set_random_seed(seed)

    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    env_cfg = cfg.env
    logger.warning("Removing observation uncertainty")
    env_cfg["obs_uncertainty"]["handle:pose/position"]["std"] = 0.0
    preconds = pkl_load(cfg.path_to_preconds, True)
    task_set = cfg.task_set
    logger.info(f"Task Set {task_set}")
    logger.info("-----------------")

    if cfg.mode == "train":
        if cfg.train.use_ray:
            ray.init(num_cpus=cfg.train.num_cpus + 2)

        nominal_skill_chain = load_nominal_skill_chain(cfg)
        # failures = pkl_load(cfg.path_to_failures, True)
        failure_clusters = pkl_load(
            cfg.path_to_failure_clusters,
            True,
        )
        failure_clusters = [cluster for cluster in failure_clusters if
                            len(cluster) >= 10]
        logger.info(f"Solving {len(failure_clusters)} failure clusters")
        train_tasks = pkl_load(join(
            cfg.path_to_tasks_root, f"train_tasks_{task_set}.pkl"), True)
        val_tasks = pkl_load(join(
            cfg.path_to_tasks_root, f"val_tasks_{task_set}.pkl"), True)
        subgoals = pkl_load(cfg.path_to_preconds, True)

        updated_skill_chain = nominal_skill_chain

        if cfg.planner == "Sequential":
            recovery_skills = solve_failures_sequentially(
                failure_clusters, updated_skill_chain, subgoals, cfg
            )
        elif cfg.planner == 'Weighted-Round-Robin':
            recovery_skills = solve_failures_round_robin(
                train_tasks, val_tasks, updated_skill_chain, subgoals,
                weighted=True, cfg=cfg
            )
        elif cfg.planner == 'Iterative-Round-Robin':
            recovery_skills = solve_failures_iteratively(
                train_tasks, val_tasks, updated_skill_chain, subgoals, cfg,
            )
        elif cfg.planner == 'Round-Robin':
            recovery_skills = solve_failures(
                train_tasks, val_tasks, updated_skill_chain, subgoals, cfg,
            )
        elif cfg.planner == 'Transition-UCB':
            recovery_skills = solve_failures(
                train_tasks, val_tasks, updated_skill_chain, subgoals, cfg,
            )
        else:
            raise NotImplementedError


    if cfg.mode == "init":
        # Initialize with a segmented demo
        env = FrankaDoorEnv(
            controller_configs=controller_cfg,
            has_renderer=True,
            horizon=cfg.env.horizon,
        )

        for i in range(cfg.ndemos):
            logger.info(f"Recording {i}th demo")
            env.reset()
            demo = record_demo(env, cfg, controller_cfg, seed)
            actions = demo["actions"]
            timesteps = [1] * len(actions)
            good = input("Is the demo good? (y/n)")
            good = 1 if good == "y" else 0

            if good:
                # segment into a skill chain
                skill_chain = segment_demo(demo)
                # record_relevant_vars(skill_chain, env)
                pkl.dump(skill_chain, open(f"skill_chain_{i}.pkl", "wb"))

            logger.info(f"Saving {i}th demo")
            pkl.dump(
                dict(quality=good, demo_results=demo),
                open(f"{good}_demo_results_{i}.pkl", "wb"),
            )

    elif cfg.mode == "eval":
        logger.info("Evaluating policy")

        if cfg.policy_type == "demo":
            env = FrankaDoorEnv(
                controller_configs=controller_cfg,
                has_renderer=cfg.render,
                horizon=cfg.env.horizon,
            )
            env.reset_from_state(goal_state)
            skill = IdentityEESpaceSkill(
                path_to_policy=to_absolute_path(cfg.path_to_policy),
            )

            # skill_db.add_skill(skill)
            skill.apply(env, env.obs(), None, render=cfg.render)

        elif cfg.policy_type == "REPS":
            env_cfg["obs_uncertainty"]["handle:pose/position"]["std"] = 0.0
            skills = pkl_load(cfg.path_to_recovery_skills, True)
            eval_env = VisualizationWrapper(
                FrankaDoorSubtaskEnv(
                    # FrankaDoorSubtaskEnv(
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
            # keys=cfg.env.obs_vars,
            failures = pkl_load(cfg.path_to_failure_clusters, True)
            preconds = pkl_load(cfg.path_to_preconds, True)

            cluster_id = 5
            subgoal_id = 3 #2 # 1
            knn_skill = True
            # knn_skill = False
            generate_failure = False
            # generate_failure = True
            # starts = iter(failures[cluster_id])
            success = 0
            nfails = 0

            while True:
                if isinstance(skills[cluster_id][subgoal_id], dict):
                    # _skills = [skill for skill in skills[cluster_id] if skill is not None]
                    # _skills = [skills[cluster_id][subgoal_id]['skill']]
                    skill = skills[cluster_id][subgoal_id]['skill']
                    skill = NearestNeighborSkill(skill.skills,
                                                 # skill.goal_constraint,
                                                 preconds[subgoal_id],
                                                 skill.env_cfg)
                    ground_skills = skill.skills
                    knn_skill = True
                else:
                    # _skills = np.concatenate((skills[cluster_id][subgoal_id]))
                    _skills = skills[cluster_id][subgoal_id]
                # failure = np.random.choice(failures[cluster_id])
                # for i, skill in enumerate(_skills):
                # for i in range(cfg.nevals):
                i = 0
                # while nfails < cfg.nevals:
                for ground_skill in  ground_skills:
                    # logger.info(f"Skill {i}:")
                    if knn_skill:
                        pass
                        # train_skills = skill.skills
                        # low_level_skill = np.random.choice(train_skills)
                        # skill = low_level_skill
                        # start = low_level_skill.start
                        # start = failure
                    else:
                        # start = skill.start
                        # print(start['hinge:pose/theta'])
                        pass
                        # start = next(starts)
                    # goal = ground_skill.goal
                    goal = skill.goal_constraint
                    eval_env.env.goal_constraint = goal
                    eval_env.set_obs_corruption(False)

                    obs = eval_env.reset()

                    if generate_failure:
                        noise = np.random.normal(0, 0.04, size=3)
                        obs['handle:pose/position'] += noise
                        logger.info(f"   noise: {noise}")
                        nominal_skill = ReachAndGraspHandleSkill()
                        obs, rew, done, info = nominal_skill.apply(eval_env, obs, render=cfg.render)
                        all_precond_sat = [#preconds[0].is_satisfied(obs),
                                           preconds[1].is_satisfied(obs),
                                           preconds[-1].is_satisfied(obs)]
                        # if info['is_solved']:
                        if any(all_precond_sat):
                            logger.info("  Nominal skill worked")
                            continue
                        else:
                            nfails += 1
                            logger.info("  Nominal skill failed")

                    else:
                        if knn_skill:
                            failure = failures[cluster_id][i]
                        else:
                            failure = skill.start
                        obs = eval_env.reset_from_state(failure)
                        nfails += 1

                    logger.info("  Executing recovery")
                    obs = eval_env.obs(ground_truth=True)
                    obs, rew, done, info = skill.apply(
                        eval_env, obs, render=cfg.render, deterministic=True,
                        # local_frame=True
                    )
                    # __import__('ipdb').set_trace()
                    if info['is_solved']:
                        success += 1
                    logger.info(f"  Reward: {rew}")
                    # logger.info(
                        # f"  Distance: {skill.goal_constraint.distance(eval_env.state())}"
                    # )
                    logger.info(f"  Solved: {info['is_solved']}")
                    i += 1
                logger.info(f"Recovery Rate: {success / nfails * 100 }%")
                return



if __name__ == "__main__":
    main()
