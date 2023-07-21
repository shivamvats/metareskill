import hydra
from hydra.utils import to_absolute_path
import ray

from recovery_skills.utils import *
from recovery_skills.skills import REPSSkill


@hydra.main(config_path="../cfg", config_name="learn_recovery_skills.yaml")
def main(cfg):
    ray.init(num_cpus=cfg.train.num_cpus)
    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    recovery_tasks = pkl_load(
        "data/door_opening/debug/failures/12-May/recovery_tasks.pkl", True
    )
    skills = []

    for i, task in enumerate(recovery_tasks):
        logger.info(f"Solving recovery task {i}")
        logger.info("--------------------------")

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
        skills.append((skill, info['history']['solved_frac'][-1][-1]))
        pkl_dump(skills, 'recovery_skills.pkl')

        if solved:
            logger.info("  Success")
        else:
            logger.info("  Failure")


if __name__ == "__main__":
    main()
