import logging

from gym.spaces import Box
from icecream import ic
import numpy as np
from recovery_skills.skills.perturbation_skill import PerturbationSkill

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class SimpleStateEstimation:
    """Rejects particles based on handle grasp success."""

    def __init__(self):
        self.sensing_skill = PerturbationSkill()

        handle_length = 0.1
        self.ee_box_space = Box(
            low=np.array([-handle_length - 0.009, -0.14, -0.04]),
            # low=np.array([-0.009, -0.14, -0.04]),
            # high=np.array([0.009, 0.04, 0.04])
            high=np.array([0.009, 0.01, 0.04]) # ee travels from -y to +y
        )

    def set_prior(self, particles):
        """Prior is a set of particles along with the corresponding weights."""

        self.particles = np.array(particles)
        self.weights = np.ones(len(particles))

    def update(self, handle_grasped, ee_pos):
        """The filter assumes the handle is at different locations and executes
        the same sensing action in real world and with particles compares the
        outcomes.
        """

        self.measurement_update(handle_grasped, ee_pos)
        # self.resample()

        logger.info(f"MLE: {self.mle}")

    def measurement_update(self, handle_grasped, ee_pos):
        """Update the weights of particles based on the observations."""

        # Can only trust the robot measurements
        if handle_grasped:
            probs = np.ones_like(self.weights)
            filtered_ids = []
            for i, particle in enumerate(self.particles):
                if not self.ee_box_space.contains(np.array(particle) - ee_pos):
                    probs[i] = 1e-8
                    filtered_ids.append(i)
            logger.debug(f"{len(filtered_ids)} particles filtered")
            self.weights[:] *= probs
            self.weights /= np.sum(self.weights)
            # for simplicity
            # I could also compute_mle
            self.mle = ee_pos

        else:
            probs = np.ones_like(self.weights)
            filtered_ids = []
            for i, particle in enumerate(self.particles):
                if self.ee_box_space.contains(np.array(particle) - ee_pos):
                    probs[i] = 1e-8
                    filtered_ids.append(i)
            logger.debug(f"{len(filtered_ids)} particles filtered")

            self.weights[:] *= probs
            # renormalize
            self.weights /= np.sum(self.weights)
            self.mle = self.compute_mle()

    def _dist_to_prob(self, dist, weight=100):
        prob = np.exp(-weight * dist)

        return prob

    # def resample(self):
        # particle_idxs = np.arange(self.num_particles)
        # sampled_idxs = np.random.choice(particle_idxs, size=len(particle_idxs), p=probs)
        # sampled_particles = self.particles[sampled_idxs]

        # self.particles = sampled_particles

        # ic(self.weights)
        # # sample with replacement and set all weights to 1
        # self.weights[:] = 1.0

    def compute_mle(self):
        """Return the most likely particle"""

        best_prob = np.max(self.weights)
        # check if there are particles with similar prob
        lower_bound = best_prob - 1e-4
        best_particles = self.particles[self.weights > lower_bound]
        return best_particles[np.random.randint(len(best_particles))]

    def visualize_mle(self, env, render_steps=10):

        env.reset_handle_pose(self.mle)
        env.set_indicator_pos('indicator0', self.mle)
        env.sim.forward()
        env.sim.step()
        for _ in range(render_steps):
            env.render()

    def visualize_particles(self, env):
        for i in range(self.num_particles):
            env.set_indicator_pos(f'indicator{i}', [0, 0, 0])
            if self.weights[i] > 0.01:
                env.set_indicator_pos(f'indicator{i}', self.particles[i])
        env.sim.step()

    @property
    def num_particles(self):
        return len(self.particles)
