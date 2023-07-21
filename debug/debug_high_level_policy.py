import hydra
from hydra.utils import to_absolute_path
import logging
import ray
from sklearn.model_selection import *

from robosuite.wrappers import VisualizationWrapper
from recovery_skills.utils import *
from recovery_skills.recovery.recovery_task import *
from recovery_skills.skills.nearest_neighbor_skill import *
from recovery_skills.envs.franka_door_subtask_env import FrankaDoorSubtaskEnv

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


@hydra.main(config_path="../cfg", config_name="learn_recovery_skills.yaml")
def main(cfg):
    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    env_cfg = cfg.env
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
    env.set_obs_corruption(False)
    path_to_recovery_skills = 'data/door_opening/debug/recovery_skills/14-May/recovery_skills.pkl'

    recovery_skills = pkl_load(path_to_recovery_skills, True)
    skills = recovery_skills[0][1]
    tasks = [RecoveryTask(skill.start, skill.goal, 1) for skill in skills]
    tasks_train, tasks_test, skills_train, skills_test = train_test_split(
        tasks, skills, test_size=0.2)
    goal = tasks[0].goal_constraint
    knn_skill = NearestNeighborSkill(skills_train,
                                    goal,
                                    env_cfg)

    env.set_goal_constraint(goal)
    obs = env.reset()

    successes = []
    n_neighbors = np.arange(1, 10)
    for n_neighbor in n_neighbors:
        logger.info(f"n_neighbor: {n_neighbor}")
        logger.info("-------------------")

        knn_skill.skill_regression_model = knn_skill._train_nearest_neighbor_regressor(
            skills_train, n_neighbors=n_neighbor)
        success = 0
        for task in tasks_test:
            obs = env.reset_from_state(task.start)
            obs, rew, done, info = knn_skill.apply(
                env, obs, render=cfg.render, deterministic=True
            )
            if info['is_solved']:
                success += 1
            print(f"  Reward: {rew}")
            print(
                f"  Distance: {knn_skill.goal_constraint.distance(env.state())}"
            )
            print(f"  Solved: {info['is_solved']}")
        successes.append(success)
        logger.info(f"  Success rate: {success/len(tasks_test)}")
        pkl_dump(successes, "successes.pkl")
    successes = np.array(successes)
    logger.info(f"n_neighbors: {n_neighbors}")
    logger.info(f"Successes: {successes/len(tasks_test)}")

if __name__ == "__main__":
    main()
