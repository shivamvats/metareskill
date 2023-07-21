"""
XXX
Since we assume the MLE to be correct, transition probabilities should be
evaluated under 0 uncertainty. This implies that the nominal skills would work
perfectly!
"""

import hydra
import logging
import numpy as np
from tqdm import trange

from recovery_skills.envs import *
from recovery_skills.skills.nominal_door_opening_skills import *
from recovery_skills.utils import *
from recovery_skills.graph.preconditions import *
from recovery_skills.skills.utils import *
from recovery_skills.graph.symbolic_graph import *

logger = logging.getLogger(__name__)


@hydra.main(config_path="../cfg", config_name="estimate_transition_probs.yaml")
def main(cfg):
    controller_cfg = OmegaConf.to_container(cfg.env.controller)
    env_cfg = cfg.env

    if cfg.uncertainty_level == 'high':
        env_cfg['obs_uncertainty']['handle:pose/position']['std'] = 0.02
    elif cfg.uncertainty_level == 'low':
        env_cfg['obs_uncertainty']['handle:pose/position']['std'] = 0.005
    else:
        raise NotImplementedError

    env = FrankaDoorEnv(
            controller_configs=controller_cfg,
            has_renderer=cfg.render,
            horizon=env_cfg.horizon,
            context_cfg=env_cfg.context,
            obs_uncertainty=env_cfg.obs_uncertainty,
            timestep=env_cfg.timestep,
            eef_start_region_cfg=env_cfg.eef_start_region,
            )
    env.set_obs_corruption(True)

    skills = [ReachAndGraspHandleSkill(),
              RotateHandleSkill(),
              PullHandleSkill()]
    skill_chain = load_nominal_skill_chain(cfg)
    preconds = pkl_load(cfg.path_to_preconds, True)
    failure_clf = pkl_load(cfg.path_to_failure_classifier, True)
    graph = SymbolicGraph(preconds, failure_clf)

    nsubgoals = len(preconds)
    nfailure_clusters = sum(failure_clf.nclusters)
    nstates = nsubgoals + nfailure_clusters + 1 # subgoal, failure clustes, FAIL
    n_actions = len(skills)

    T = np.zeros((n_actions, nstates, nstates), dtype=int)
    probs = np.zeros_like(T, dtype=float)

    for t in trange(cfg.nevals):
        obs = env.reset()
        # env.render_by()
        pkl_dump(T, "transition_counts.pkl")
        pkl_dump(probs, "transition_probs.pkl")
        logger.info(f"Transition counts\n{T}")
        logger.info(f"Transition probs\n{probs}")

        for i, skill in enumerate(skills):
            # assume no uncertainty in symbolic states (optimistic)
            gt_obs = env.obs(ground_truth=True)
            start_id = graph.state_to_id(gt_obs)
            # start_id = get_state_id(True, i, nstates)
            precond_satisfied = preconds[i].is_satisfied(gt_obs)
            # assume uncertainty in symbolic states (actual)
            # obs = env.obs()
            # precond_satisfied = preconds[i].is_satisfied(obs)

            if cfg.discovery_strategy == 'open-loop':
                precond_satisfied = True

            if i == 0:
                precond_satisfied = True

            if precond_satisfied:
                # logger.info(f"Preconds {i} satisfied")
                obs, rew, done, info = skill.apply(
                    env, obs, None, render=cfg.render
                )
                gt_obs = env.obs(ground_truth=True)
                if preconds[i+1].is_satisfied(gt_obs) or env._check_success(gt_obs):
                    subgoal = True
                    local_id = i+1
                else:
                    subgoal = False
                    local_id = failure_clf.predict(gt_obs)
                # goal_id = get_state_id(subgoal, local_id, nstates)
                goal_id = graph.state_to_id(gt_obs)
                T[i, start_id, goal_id] += 1
                probs = T / np.sum(T, axis=2, keepdims=True)
            else:
                break
        if not env._check_success(gt_obs):
            logger.info("  Task failed")
        else:
            logger.info("  Task succeeded")


if __name__ == "__main__":
    main()
