import hydra
from hydra.utils import to_absolute_path
import numpy as np

from recovery_skills.envs import *
from recovery_skills.utils import *
from recovery_skills.graph.preconditions import *


@hydra.main(config_path="../cfg", config_name="learn_subgoals.yaml")
def main(cfg):
    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    env_cfg = cfg.env

    env = FrankaDoorEnv(
            controller_configs=controller_cfg,
            has_renderer=cfg.render,
            horizon=env_cfg.horizon,
            context_cfg=env_cfg.context,
            obs_uncertainty=env_cfg.obs_uncertainty,
            timestep=env_cfg.timestep,
            eef_start_region_cfg=env_cfg.eef_start_region,
            )

    all_subgoals, labels = load_subgoals(cfg.subgoals_dir, ground_truth=True)
    # obs  = pkl_load(cfg.obs_file, True)
    # labels = pkl_load(cfg.label_file, True)
    # labels = np.array(labels)

    # subgoal_id = 1
    subgoal_ids = [0, 1, 2, 3]

    subgoal_clfs = []

    # skill_ids = [0]
    # for subgoal_id in range(skill_ids):
    for subgoal_id in subgoal_ids:
        logger.info(f"Subgoal {subgoal_id}")
        logger.info("------------\n")
        subgoals = np.array(all_subgoals[subgoal_id])
        obs = subgoals

        # neg_subgoals = subgoals[np.where(np.logical_not(labels))]
        # for state in neg_subgoals:
            # env.reset_from_state(state)
            # env.render_by(100)

        subgoal = BayesPreconditionClassifier(obs, labels)
        # __import__('ipdb').set_trace()
        subgoal_clfs.append(subgoal)
        # for subgoal in subgoals:
            # rew = precond.reward(subgoal)
            # print(rew)

    pkl_dump(subgoal_clfs, 'subgoals.pkl')


if __name__ == "__main__":
    main()
