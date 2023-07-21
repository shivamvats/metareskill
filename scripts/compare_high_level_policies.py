"""Compare the behavior of different regression models as high level policy.,"""

from os.path import join

import hydra
from hydra.utils import to_absolute_path
import logging
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed

from recovery_skills.utils import *
from recovery_skills.envs.franka_door_env import FrankaDoorEnv
from recovery_skills.envs.franka_door_subtask_env import FrankaDoorSubtaskEnv
from recovery_skills.skills.nearest_neighbor_skill import *

logger = logging.getLogger(__name__)


def make_env(cfg, env_cfg):
    controller_cfg = OmegaConf.to_container(env_cfg.controller)
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

    env.set_obs_corruption(False)

    return env


def compute_skill_accuracy(hl_skill, env, tasks, render=False):
    solved = []
    for task in tasks:
        start = task.start
        goal = task.goal_constraint
        obs = env.reset_from_state(start)
        env.set_goal_constraint(goal)
        obs, rew, done, info = hl_skill.apply(env, obs, None, render=render)
        solved.append(info['is_solved'])

    return np.mean(solved)


@hydra.main(config_path="../cfg", config_name="evaluate_recovery_skills")
def main(cfg):
    if cfg.seed is None:
        seed = int(time.time())
    else:
        seed = cfg.seed
    if cfg.task_set is not None:
        seeds = [385, 52, 894, 99, 28]
        seed = seeds[cfg.task_set]
    set_random_seed(seed)

    env_cfg = cfg.env
    env = make_env(cfg, env_cfg)
    task_set = cfg.task_set
    if task_set is None:
        task_set = 0

    preconds = pkl_load(cfg.path_to_preconds, True)
    # path_to_skill_dir = "data/door_opening/debug/recovery_skills/8-Sep/rr/budget-300/"
    # path_to_skill_dir = "data/door_opening/debug/recovery_skills/9-Sep/rr/budget-500/"
    path_to_skill_dir = "data/door_opening/debug/recovery_skills/13-Sep/rr/0/"
    recovery_skills = pkl_load(join(path_to_skill_dir, 'recovery_skills.pkl'),
                               True)
    failure_clf = pkl_load(cfg.path_to_failure_clf, True)
    train_tasks = pkl_load(join(
        cfg.path_to_tasks_root, f"train_tasks_{task_set}.pkl"), True)
    val_tasks = pkl_load(join(
        cfg.path_to_tasks_root, f"val_tasks_{task_set}.pkl"), True)

    n_data = []
    for row in recovery_skills:
        data = []
        for skills in row:
            data.append(len(skills))
        n_data.append(data)

    n_data = np.array(n_data)

    fig, ax = plt.subplots()
    ax.imshow(n_data)
    for (j, i), label in np.ndenumerate(n_data):
        ax.text(i, j, label, ha='center', va='center')
    ax.set_title("Data points")
    ax.set_xlabel("Subgoals")
    ax.set_ylabel("Failure Clusters")
    fig.savefig("num_data.png")
    plt.close(fig)


    ##
    # Evaluation
    ##

    sorted_idx_i, sorted_idx_j = np.unravel_index(np.argsort(-n_data, axis=None),
                                                  n_data.shape,)


    fig, ax = plt.subplots()
    model = "lin_reg"
    logger.info("Linear Regression")
    logger.info("==============\n")
    for i, j in zip(sorted_idx_i[:5], sorted_idx_j[:5]):
        skills = recovery_skills[i][j]
        goal = preconds[j]

        logger.info(f"Analyzing policy id: {i, j} with #data {len(skills)}")
        n_train_data, accs = [], []
        for k in range(1, len(skills)):
            train_skills = skills[:k]
            hl_policy = NearestNeighborSkill(train_skills, goal, env_cfg)
            hl_policy.skill_regression_model = hl_policy._train_linear_regressor(hl_policy.skills)
            tasks = val_tasks[i][j]
            acc = compute_skill_accuracy(hl_policy, env, tasks, render=cfg.render)
            n_train_data.append(k), accs.append(acc)

            logger.info("Results\n")
            logger.info("---------")
            logger.info(f"  # Training Points: {len(train_skills)}")
            logger.info(f"  # Evals: {len(tasks)}")
            logger.info(f"  Accuracy: {acc}")

        ax.plot(n_train_data, accs, label=f"{i, j}")
        ax.set_title(f"{model}")
        ax.set_xlabel("# data points")
        ax.set_ylabel("Accuracy")
        fig.savefig(f"acc_vs_data_{model}.png")
        ax.legend()
    fig.savefig(f"acc_vs_data_{model}.png")
    plt.close()

if __name__ == "__main__":
    main()
