import logging

from icecream import ic
import numpy as np
from recovery_skills.skills.perturbation_skill import PerturbationSkill

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class ParticleFilter:
    def __init__(self, env):
        self.env = env

        self.sensing_skill = PerturbationSkill()

    def set_prior(self, particles):
        """Prior is a set of particles along with the corresponding weights."""

        self.particles = np.array(particles)
        self.weights = np.ones(len(particles))

    def update_perturbation(
        self, real_world_start, real_world_start_obs, real_world_final_obs
    ):
        """The filter assumes the handle is at different locations and executes
        the same actions executed by the real world robot and compares the
        outcomes.

        The filter executes a perturbation skill to sense the contact modes.
        """

        self.perturbation = True

        self.mle = self.compute_mle()
        logger.info(f"MLE: {self.mle}")

        particle_obs = []
        for particle in self.particles:
            self.env.reset_from_state(real_world_start)
            obs = self.env.reset_handle_pose(particle)

            obs, _, _, _ = self.sensing_skill.apply(
                self.env, real_world_start_obs, context=None, render=True
            )
            particle_obs.append(obs)

        self.measurement_update(particle_obs, real_world_final_obs)
        self.resample()


    def update_skill(
        self, skill, real_world_start, real_world_start_obs, real_world_final_obs
    ):
        """The filter assumes the handle is at different locations and executes
        the same actions executed by the real world robot and compares the
        outcomes."""

        self.perturbation = False

        self.env.reset_from_state(real_world_start)

        self.mle = self.compute_mle()
        logger.info(f"MLE: {self.mle}")

        particle_obs = []
        for particle in self.particles:
            obs = self.env.reset_handle_pose(particle)
            # Execute the same actions as in real world
            obs, _, _, _ = skill.apply(
                self.env, real_world_start_obs, context=None, render=True
            )
            particle_obs.append(obs)

        self.measurement_update(particle_obs, real_world_final_obs)
        self.resample()

    def measurement_update(self, particle_obs, real_world_obs):
        """Update the weights of particles based on the observations."""

        if self.perturbation:
            # Can only trust the robot measurements
            rw_perturb = real_world_obs['perturbations']

            for i, obs in enumerate(particle_obs):
                # compare eef position
                perturb = obs['perturbations']
                ic(perturb.round(3))
                perturb_dist = np.linalg.norm(perturb - rw_perturb)

                total_dist = perturb_dist
                self.weights[i] = self._dist_to_prob(total_dist)

        else:

            # Can only trust the robot measurements
            gt_eef_pos = real_world_obs["robot_eef:pose/position"]
            gt_gripper_pos = real_world_obs["robot_eef:gripper/position"]

            for i, obs in enumerate(particle_obs):
                # compare eef position
                eef_pos = obs["robot_eef:pose/position"]
                eef_dist = np.linalg.norm(eef_pos - gt_eef_pos)

                # compare gripper position
                gripper_width_dist = 2 * np.abs(
                    obs["robot_eef:gripper/position"] - gt_gripper_pos
                )

                logger.debug(
                    f"Errors: EEF({eef_dist:.2f}); Gripper ({gripper_width_dist:.2f})"
                )

                total_dist = gripper_width_dist + 0.1 * eef_dist
                self.weights[i] = self._dist_to_prob(total_dist)

    def _dist_to_prob(self, dist):
        if self.perturbation:
            prob = np.exp(-dist)
        else:
            prob = np.exp(-100 * dist)

        return prob

    def resample(self):
        probs = self.weights / np.sum(self.weights)
        particle_idxs = np.arange(len(self.particles))
        sampled_idxs = np.random.choice(particle_idxs, size=len(particle_idxs), p=probs)
        sampled_particles = self.particles[sampled_idxs]

        self.particles = sampled_particles

        ic(self.weights)
        # sample with replacement and set all weights to 1
        self.weights[:] = 1.0

    def compute_mle(self):
        """Return the most likely particle"""

        return self.particles[np.argmax(self.weights)]
