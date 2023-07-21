import numpy as np
from recovery_skills.skills.perturbation_skill import PerturbationSkill


def check_handle_grasped(env):
    # Perturb normal to the door
    skill = PerturbationSkill(axes=['y'])
    obs = env.obs()
    obs, _, _, _ = skill.apply(env, obs, None, render=True)

    perturbs = obs['perturbations']

    # Along each axis, check if eef can more in neigher +- directions
    thresh = 0.8 # 0.8 * 2 = 1.6 cm

    perturbs = np.abs(perturbs)
    max_perturbs = np.max(perturbs, axis=1)
    # logger.debug(f"Maximum perturbations: {max_perturbs}")

    can_move_along_axes = max_perturbs > thresh

    is_free = any(can_move_along_axes)
    is_grasped = not(is_free)

    return is_grasped
