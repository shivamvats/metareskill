import copy
import numpy as np
from recovery_skills.graph import State
from collections import OrderedDict

import ray


class RayVecEnvWrapper():
    """Takes in a list of ray actor environments and returns concatenated
    results from them."""
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self):
        obsvs = [env.reset.remote() for env in self.envs]
        obsvs = ray.get(obsvs)
        return np.array(obsvs)

    def reset_from_state(self, state):
        if isinstance(state, State) or isinstance(state, OrderedDict):
            states = [copy.deepcopy(state) for _ in range(self.num_envs)]
        else:
            states = state
            assert len(states) == self.num_envs
        obsvs = [env.reset_from_state.remote(state) for env, state in
                 zip(self.envs, states)]
        obsvs = ray.get(obsvs)

        return np.array(obsvs)

    def step(self, actions):
        results = [env.step.remote(action) for env, action in zip(self.envs,
                                                                  actions)]
        results = ray.get(results)

        obsvs = [result[0] for result in results]
        rewards = [result[1] for result in results]
        dones = [result[2] for result in results]
        infos = [result[3] for result in results]
        return obsvs, rewards, dones, infos

    def render(self):
        results = [env.render.remote() for env in self.envs]
        ray.get(results)

    def seed(self):
        results = [env.seed.remote() for env in self.envs]
        results = ray.get(results)
        return results

    def state(self):
        states = [env.state.remote() for env in self.envs]
        states = np.array(ray.get(states))
        return states

    def rl_state(self):
        rl_states = [env.rl_state.remote() for env in self.envs]
        rl_states = np.array(ray.get(rl_states))
        return rl_states

    def context(self):
        contexts = [env.context.remote() for env in self.envs]
        contexts = np.array(ray.get(contexts))
        return contexts

    def unflatten_obs(self, flat_obs):
        obs_dicts = [env.unflatten_obs.remote(obs) for obs, env in zip(flat_obs,
                                                               self.envs)]
        obs_dicts = ray.get(obs_dicts)
        return obs_dicts

    def get_warm_initialization(self, *args, **kwargs):
        # return self.envs[env_idx].get_warm_initialization(**kwargs)
        # All envs are supposed to be copies of the same problem
        result = self.envs[0].get_warm_initialization.remote(*args, **kwargs)
        result = ray.get(result)
        return result

    def set_goal_constraint(self, goal):
        results = [env.set_goal_constraint.remote(goal) for env in self.envs]
        results = ray.get(results)
        return results
