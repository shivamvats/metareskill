from collections import OrderedDict
from numbers import Number

from gym import spaces
import numpy as np
import ray

from .franka_door_env import FrankaDoorEnv
from .franka_door_subtask_env import FrankaDoorSubtaskEnv


@ray.remote(num_cpus=0.5, num_gpus=0.05)
class RayActorWrapper(FrankaDoorEnv):
    """Gym wrapper that inherits `FrankaDoor` so I can use it as a ray actor."""

    def __init__(self, keys, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set up observation and action spaces
        self.keys = keys

        obs = super().reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        # flat_ob = self.flatten_obs(obs)
        # self.obs_dim = len(flat_ob)
        # high = np.inf * np.ones(self.obs_dim)
        # low = -high
        # self.observation_space = spaces.Box(low=low, high=high)
        # low, high = self.action_spec
        # self.action_space = spaces.Box(low=low, high=high)

    # @ray.method(num_returns=4)
    def step(self, action):
        ob_dict, reward, done, info = super().step(action)
        return self.flatten_obs(ob_dict), reward, done, info

    def flatten_obs(self, obs, verbose=False):
        # return self._flatten_obs(obs, verbose)
        return obs

    def unflatten_obs(self, flat_obs):
        return flat_obs
        # obs = OrderedDict()
        # t = 0
        # for key, dim in zip(self._obs_keys, self._obs_dims):
            # value = flat_obs[t:t+dim]
            # if dim == 1:
                # obs[key] = np.array(value)
            # else:
                # obs[key] = np.array(value)
            # t += dim
        # return obs

    def render(self, *args, **kwargs):
        return super().render()

    def reset(self, **kwargs):
        ob_dict = super().reset(**kwargs)
        return self.flatten_obs(ob_dict)

    def reset_from_state(self, state, **kwargs):
        ob_dict = super().reset_from_state(state, **kwargs)
        return self.flatten_obs(ob_dict)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return super().reward()

    def apply(self, policy, obs, **kwargs):
        return policy.apply(self, obs, **kwargs)

    def _flatten_obs(self, obs_dict, verbose=False):
        ob_lst = []
        self._obs_keys, self._obs_dims = [], []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())

                # Book-keeping for unflattening
                self._obs_keys.append(key)
                value = obs_dict[key]
                if isinstance(value, Number) or np.array(value).ndim == 0:
                    self._obs_dims.append(1)
                else:
                    self._obs_dims.append(len(value))
        return np.concatenate(ob_lst)


@ray.remote(num_cpus=0.5, num_gpus=0.05)
class RayActorSubtaskWrapper(FrankaDoorSubtaskEnv):
    """Gym wrapper that inherits `FrankaDoor` so I can use it as a ray actor."""

    def __init__(self, keys=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set up observation and action spaces
        self.keys = keys

        # obs = super().reset()
        # self.modality_dims = {key: obs[key].shape for key in self.keys}
        # flat_ob = self.flatten_obs(obs)
        # self.obs_dim = flat_ob.size
        # high = np.inf * np.ones(self.obs_dim)
        # low = -high
        # self.observation_space = spaces.Box(low=low, high=high)
        # low, high = self.action_spec
        # self.action_space = spaces.Box(low=low, high=high)

    # @ray.method(num_returns=4)
    def step(self, action):
        ob_dict, reward, done, info = super().step(action)
        return self.flatten_obs(ob_dict), reward, done, info

    def flatten_obs(self, obs, verbose=False):
        # return self._flatten_obs(obs, verbose)
        return obs

    def unflatten_obs(self, flat_obs):
        return flat_obs
        # obs = OrderedDict()
        # t = 0
        # for key, dim in zip(self._obs_keys, self._obs_dims):
            # value = flat_obs[t:t+dim]
            # if dim == 1:
                # obs[key] = np.array(value)
            # else:
                # obs[key] = np.array(value)
            # t += dim
        # return obs

    def render(self, *args, **kwargs):
        return super().render()

    def reset(self,  **kwargs):
        ob_dict = super().reset(**kwargs)
        return self.flatten_obs(ob_dict)

    def reset_from_state(self, state, **kwargs):
        ob_dict = super().reset_from_state(state, **kwargs)
        return self.flatten_obs(ob_dict)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return super().reward()

    def apply(self, policy, obs, **kwargs):
        return policy.apply(self, obs, **kwargs)

    def _flatten_obs(self, obs_dict, verbose=False):
        ob_lst = []
        self._obs_keys, self._obs_dims = [], []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())

                # Book-keeping for unflattening
                self._obs_keys.append(key)
                value = obs_dict[key]
                if isinstance(value, Number) or np.array(value).ndim == 0:
                    self._obs_dims.append(1)
                else:
                    self._obs_dims.append(len(value))
        return np.concatenate(ob_lst)
