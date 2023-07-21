from copy import deepcopy
import logging
import hydra
from hydra.utils import to_absolute_path
from icecream import ic
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed
from tqdm import trange, tqdm

from recovery_skills.envs.improved_gym_wrapper import ImprovedGymWrapper
from recovery_skills.envs.franka_door_env import FrankaDoorEnv
from recovery_skills.envs.franka_door_subtask_env import FrankaDoorSubtaskEnv
from recovery_skills.skills import SkillChain
from recovery_skills.skills.nominal_door_opening_skills import (
# from recovery_skills.skills.robust_door_opening_skills import (
    # ReachHandleSkill,
    # GraspHandleSkill,
    ReachAndGraspHandleSkill,
    RotateHandleSkill,
    PullHandleSkill,
)
from recovery_skills.skills.nearest_neighbor_skill import NearestNeighborSkill
from recovery_skills.graph.preconditions import (BayesPreconditionClassifier,
                                                 DoorOpenStartPrecondition,
                                                 DoorOpenGoalPrecondition)
from recovery_skills.utils import *

logger = logging.getLogger(__name__)


def make_env(cfg, env_cfg):
    controller_cfg = OmegaConf.to_container(env_cfg.controller)

    if cfg.train.use_ray:
        raise NotImplementedError

    else:
        # env = FrankaDoorEnv(
        env = FrankaDoorSubtaskEnv(
                goal_constraint=None,
                controller_configs=controller_cfg,
                has_renderer=cfg.render,
                horizon=env_cfg.horizon,
                context_cfg=env_cfg.context,
                obs_uncertainty=env_cfg.obs_uncertainty,
                timestep=env_cfg.timestep,
                eef_start_region_cfg=env_cfg.eef_start_region,
            )
        if cfg.perfect_state_estimation:
            env.set_obs_corruption(False)

    return env


def learn_skill_chain_subgoals(skill_chain, env, cfg):
    """
    Learn subgoal distribution using only successful executions.

    Inputs
    ------
    * skill_chain

    Outputs
    -------
    * `init_sets.pkl` + `rl_init_sets.pkl` + 'gt_rl_init_sets'
    * `trained_nominal_skill_chain.pkl`

    """

    nskills = skill_chain.size

    # collect positive executions to estimate subgoal distribution

    all_results = []

    all_subgoals = [[] for _ in range(nskills + 1)]
    all_gt_subgoals = [[] for _ in range(nskills + 1)]
    all_labels = []
    results = []
    num_success = 0

    for i in trange(cfg.nevals):
        obs = env.reset()
        obs, rew, done, info = skill_chain.apply(env,
        # obs, rew, done, info = skill_chain.apply_with_state_estimation(env,
                                                 obs,
                                                 context=None,
                                                 precond_check=False,
                                                 render=cfg.render)
        results.append(info)

        task_success = env._check_success()

        if task_success:
            num_success += 1
            label = 1.0
        else:
            label = 0.0

        subgoals = info["subgoals"]
        gt_subgoals = info["gt_subgoals"]

        for j in range(len(subgoals)):
            all_subgoals[j].append(subgoals[j])
            all_gt_subgoals[j].append(gt_subgoals[j])
        all_labels.append(label)

        logger.info(f"[{i}] Task Success: {task_success}; Success Rate: {num_success} / {i+1} = {num_success/(i+1)*100} %")

        pkl_dump(all_subgoals, f"all_subgoals.pkl")
        pkl_dump(all_gt_subgoals, f"all_gt_subgoals.pkl")
        pkl_dump(all_labels, f"all_labels.pkl")

    all_results.append(results)
    pkl_dump(all_results, "all_results.pkl")

    if num_success > 2:
        pass
        # for i, skill in enumerate(skill_chain.skills):
            # y = np.zeros(len(all_states))
            # corresponding to skill i
            # y[np.arange(i * num_success, (i + 1) * num_success)] = 1
            # skill.train_precondition(all_states, all_rl_states, y)

    else:
        logger.warning(f"Not enough successful executions of skill chain")

    # pkl_dump(skill_chain, f"trained_skill_chain.pkl")
    # all_init_sets = {
        # 'init_sets': init_sets,
        # 'rl_init_sets': rl_init_sets,
        # 'gt_init_sets': gt_init_sets,
        # 'gt_rl_init_sets': gt_rl_init_sets
    # }
    # return (skill_chain, all_init_sets)
    return all_subgoals, all_gt_subgoals


def learn_skill_chain_preconds_with_chaining(skill_chain, env, init_subgoals,
                                            cfg,
                                             learnt_preconds=None,
                                             learnt_gt_preconds=None):
    """
    Train the preconditions of  skill chain

    Inputs
    ------
    * skill_chain

    Outputs
    -------
    * `init_sets.pkl` + `rl_init_sets.pkl` + 'gt_rl_init_sets'
    * `trained_nominal_skill_chain.pkl`

    """

    skills = skill_chain.skills
    nskills = skill_chain.size

    is_terminal_skill = lambda x: x == (nskills - 1)

    # forward executions to collect positive samples for preconditions
    # initiation sets of other skills are treated as negative samples for
    # SVM
    # -------

    all_results = []

    # The last one will always be None as we want to use the task success
    # condition for the last skill

    # to be used for computing value function
    # TODO Load from file
    if not learnt_preconds:
        learnt_preconds = [None]*(nskills + 1)
        learnt_gt_preconds = [None]*(nskills + 1)
    # to be used in chaining

    for skill_id in cfg.skill_ids:
        logger.info(f"Learning preconditions for skill: {skill_id}")

        skill = skills[skill_id]

        start_dist = init_subgoals[skill_id]
        # __import__('ipdb').set_trace()
        start_states = []
        for scale in [1.0, 2.0, 5.0]: #, 'uniform']:
            _start_states = start_dist.sample(cls=FrankaDoorEnv, n=int(cfg.nevals/3),
                                            env=env, scale=scale)
            start_states.append(_start_states)
        start_states = np.concatenate(start_states)

        # for state in start_states:
            # env.reset_from_state(state)
            # env.render_by(50)

        # Learnt in previous iteration
        goal_precond = learnt_gt_preconds[skill_id + 1]
        env.set_goal_constraint(goal_precond)

        subgoals = []
        gt_subgoals = []
        goal_gt_subgoals = []
        labels = []
        goal_labels = []
        results = []
        num_success = 0

        success_probs = []

        # for i in trange(cfg.nevals):
        for i, start in enumerate(start_states):
            # sample start state from the subgoal

            obs = env.reset_from_state(start)
            # env.render_by()
            subgoal = obs
            gt_subgoal = env.obs(ground_truth=True)
            # __import__('ipdb').set_trace()
            subgoals.append(subgoal)
            gt_subgoals.append(gt_subgoal)

            # env.render_by()

            # with noise
            obs, rew, done, info = skill.apply(env, obs, context=None, render=cfg.render)
            results.append(info)

            # env.render_by()

            # ground truth
            task_success = env.check_task_success()

            # terminal skill
            if is_terminal_skill(skill_id):
                _gt_obs = env.obs(ground_truth=True)
                goal_gt_subgoals.append(_gt_obs)
                goal_labels.append(task_success)

            if goal_precond:
                success_prob = goal_precond.prob(info['gt_obs'])
                success = goal_precond.is_satisfied(info['gt_obs'])
                success_probs.append(success_prob)
            else:
                success = task_success
                success_probs.append(task_success)

            if success:
                num_success += 1
                label = 1.0
            else:
                label = 0.0
            labels.append(label)

            logger.info(f"[{i}] Skill Success: {label}; Success Rate: {num_success} / {i+1} = {num_success/(i+1)*100} %")

            pkl_dump(subgoals, f"subgoals_obs_{skill_id}.pkl")
            pkl_dump(gt_subgoals, f"gt_subgoals_obs_{skill_id}.pkl")
            pkl_dump(labels, f"labels_{skill_id}.pkl")

            if is_terminal_skill(skill_id):
                pkl_dump(goal_gt_subgoals, f"gt_subgoals_obs_{skill_id + 1}.pkl")
                pkl_dump(goal_labels, f"labels_{skill_id + 1}.pkl")

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.bar(np.arange(len(gt_labels)), gt_labels, color='r', alpha=0.3)
        # ax.bar(np.arange(len(success_probs)), success_probs, color='b', alpha=0.3)
        # fig.savefig(f"task_success_vs_predicted_probs_{skill_id}.png")

        if num_success > 2:
            # Learn preconditions
            if skill_id == 0 :
                # Use hard-coded preconditions for start
                precond = DoorOpenStartPrecondition(subgoals, labels, cfg.env)
                gt_precond = DoorOpenStartPrecondition(gt_subgoals, labels,
                                                       cfg.env)
            else:
                precond = BayesPreconditionClassifier(subgoals, labels)
                gt_precond = BayesPreconditionClassifier(gt_subgoals, labels)

            if is_terminal_skill(skill_id):
                goal_gt_precond = BayesPreconditionClassifier(goal_gt_subgoals,
                                                              goal_labels)
                learnt_gt_preconds[skill_id + 1] = goal_gt_precond

            learnt_preconds[skill_id] = precond
            learnt_gt_preconds[skill_id] = gt_precond

            pkl_dump(learnt_preconds, "learnt_preconds.pkl")
            pkl_dump(learnt_gt_preconds, "learnt_gt_preconds.pkl")

        else:
            logger.warning(f"Not enough successful executions of skill chain")
        __import__('ipdb').set_trace()

    return learnt_preconds, learnt_gt_preconds


@hydra.main(config_path="../cfg", config_name="learn_skill_chain_preconditions.yaml")
def main(cfg):
    if cfg.seed is None:
        seed = int(time.time())
    else:
        seed = cfg.seed
    logger.info(f"Using seed: {seed}")
    set_random_seed(seed)

    env_cfg = cfg.env

    env = make_env(cfg, env_cfg)

    skills = [ReachAndGraspHandleSkill(),
             RotateHandleSkill(),
             PullHandleSkill()
             ]
    skill_chain = SkillChain(skills)

    if cfg.initialize:
        # initialize subgoals
        # learnt this in the sim state abstraction
        gt_subgoals, _ = learn_skill_chain_subgoals(skill_chain, env, cfg)
    else:
        # load
        gt_subgoals = pkl_load(cfg.init_subgoals, True)

    if cfg.finetune:
        # under perfect state
        env.set_obs_corruption(False)
        # finetune the initial positive samples by chaining back from the goal
        # learn this in rl state abstraction
        learnt_preconds = pkl_load(cfg.learnt_preconds, True,)
        learnt_gt_preconds = pkl_load(cfg.learnt_gt_preconds, True,)
        # learnt_preconds, learnt_gt_preconds = None, None
        # __import__('ipdb').set_trace()
        learnt_preconds, learnt_gt_preconds = learn_skill_chain_preconds_with_chaining(
            skill_chain, env, gt_subgoals, cfg, learnt_preconds, learnt_gt_preconds)

    if cfg.second_finetune:
        # add positive samples of other preconds as negative samples
        preconds = pkl_load(cfg.learnt_gt_preconds, True)

        X_pos = []

        for precond in preconds:
            x = precond.obs
            y = precond.y
            x_pos = x[y==1]
            X_pos.append(x_pos)

        # add uniformly random start states as negatives
        eef_start_region = {
                'x': {'type': 'uniform', 'range': [-0.05, 0.05]},
                'y': {'type': 'uniform', 'range': [-0.05, 0.05]},
                'z': {'type': 'uniform', 'range': [-0.05, 0.05]},
                'roll': {'type': 'uniform', 'range': [-0.8, 0.8]},
                'pitch': {'type': 'uniform', 'range': [-0.8, 0.8]},
                'yaw': {'type': 'uniform', 'range': [-0.8, 0.8]}
        }
        env_cfg['eef_start_region'] = eef_start_region

        env = make_env(cfg, env_cfg)

        random_states = []
        env.set_obs_corruption(False)
        for _ in range(50):
            obs = env.reset()
            random_states.append(obs)
            # for _ in range(25):
                # env.render()

        finetuned_preconds = []

        for i in range(len(preconds)):
            precond = preconds[i]
            obs = list(precond.obs)
            y = list(precond.y)
            obs_neg = []
            y_neg = []
            for j in range(len(preconds)):
                if i != j:
                    obs_neg += list(X_pos[j])
                    y_neg += [False]*len(X_pos[j])
            obs += obs_neg
            y += y_neg

            if i != 0:
                # sampled states are at start
                obs += random_states
                y += [False]*len(random_states)

            finetuned_precond = BayesPreconditionClassifier(obs, y)
            finetuned_preconds.append(finetuned_precond)

            pkl_dump(finetuned_preconds, "finetuned_preconds.pkl")

    elif cfg.hardcode_start_goal:
        preconds = pkl_load(cfg.learnt_gt_preconds, True)
        start = preconds[0]
        new_start = DoorOpenStartPrecondition(start.obs, start.y, env_cfg)

        goal = preconds[-1]
        new_goal = DoorOpenGoalPrecondition(goal.obs, goal.y)

        preconds[0] = new_start
        preconds[-1] = new_goal

        pkl_dump(preconds, 'final_finetuned_preconds.pkl')


if __name__ == "__main__":
    main()
