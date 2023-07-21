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

from recovery_skills.envs.improved_gym_wrapper import ImprovedGymWrapper
from recovery_skills.envs.franka_door_env import FrankaDoorEnv
from recovery_skills.envs.franka_door_subtask_env import FrankaDoorSubtaskEnv
from recovery_skills.envs import RayActorSubtaskWrapper, RayVecEnvWrapper
from recovery_skills.skills import SkillChain
from recovery_skills.skills.nearest_neighbor_skill import NearestNeighborSkill
from recovery_skills.skills.utils import *
from recovery_skills.recovery.failure_discovery import discover_failures
from recovery_skills.graph.symbolic_graph import *
from recovery_skills.utils import *
from robosuite.wrappers import VisualizationWrapper

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def make_env(cfg, env_cfg, goal_constraint):
    controller_cfg = OmegaConf.to_container(env_cfg.controller)

    if cfg.train.use_ray:
        train_envs = [
            RayActorSubtaskWrapper.remote(
                env_cfg.obs_vars,
                goal_constraint=goal_constraint,
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
        env = RayVecEnvWrapper(train_envs)

    else:
        # env = #ImprovedGymWrapper(
        # env = VisualizationWrapper(
        env = FrankaDoorSubtaskEnv(
                goal_constraint=goal_constraint,
                controller_configs=controller_cfg,
                has_renderer=cfg.render,
                horizon=env_cfg.horizon,
                context_cfg=env_cfg.context,
                obs_uncertainty=env_cfg.obs_uncertainty,
                timestep=env_cfg.timestep,
                eef_start_region_cfg=env_cfg.eef_start_region,
            )
            # 'default')
            # keys=env_cfg.obs_vars,
        # )

    return env


def viz_position(env, position):
    env.set_indicator_pos('indicator0', position)
    env.sim.step()
    for _ in range(25):
        env.render()


def verify_precond(env, obs, skill, goal_precond, **kwargs):
    env.set_obs_corruption(False)
    obs = env.reset_from_state(obs)
    obs, rew, done, info =  skill.apply(env, obs, None, **kwargs)
    success = goal_precond.is_satisfied(obs)
    # if not success:
        # __import__('ipdb').set_trace()
    return success



@hydra.main(config_path="../cfg", config_name="evaluate_nominal_skills")
def main(cfg):
    if cfg.seed is None:
        seed = int(time.time())
    else:
        seed = cfg.seed
    logger.info(f"Using seed: {seed}")
    set_random_seed(seed)

    # cfg['env']['obs_uncertainty']['handle:pose/position']['std'] = 0.04
    env_cfg = cfg.env
    skill_chain = load_nominal_skill_chain(cfg)
    preconds = pkl_load(cfg.path_to_preconds, True)
    failure_clf = pkl_load(cfg.path_to_failure_clf, True)

    nskills = skill_chain.size

    env = make_env(cfg, env_cfg, None)
    eval_env = make_env(cfg, env_cfg, None)

    sym_graph = SymbolicGraph(preconds, failure_clf)
    failures, cls_failures, _ = discover_failures(env, skill_chain, None, None, preconds,
                                 failure_clf, cfg.nevals,
                                 pessimistic_discovery=False, early_term=True,
                                 render=cfg.render)
    __import__('ipdb').set_trace()
    pkl_dump(failures, 'failures.pkl')


if __name__ == "__main__":
    main()
