from recovery_skills.envs import ImprovedGymWrapper
import robosuite as suite
from robosuite.utils.placement_samplers import ObjectPositionSampler
from stable_baselines3.common.utils import set_random_seed


def make_env(seed, env_cfg, controller_cfg, gui):
    set_random_seed(seed)
    env = suite.make(
        env_name=env_cfg.env_name,
        **env_cfg.suite,
        controller_configs=controller_cfg,
        has_renderer=gui,
    )
    env = ImprovedGymWrapper(env, keys=env_cfg.obs_keys)
    return env


# def FixedSampler(ObjectPositionSampler):
    # def __init__(self,
                 # name,
                 # mujoco_objects=None,
                 # )
