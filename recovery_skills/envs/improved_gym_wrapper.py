from collections import OrderedDict
from numbers import Number

import numpy as np
from robosuite.wrappers import GymWrapper


class ImprovedGymWrapper(GymWrapper):
    """
    Implements unflattening of flattened obs and makes flatten function public.
    """
    def flatten_obs(self, obs, verbose=False):
        return self._flatten_obs(obs, verbose)

    def unflatten_obs(self, flat_obs):
        obs = OrderedDict()
        t = 0
        for key, dim in zip(self._obs_keys, self._obs_dims):
            value = flat_obs[t:t+dim]
            if dim == 1:
                obs[key] = np.array(value)
            else:
                obs[key] = np.array(value)
            t += dim
        return obs

    def render(self, *args, **kwargs):
        """
        By default, run the normal environment render() function

        Args:
            **kwargs (dict): Any args to pass to environment render function
        """
        return self.env.render()

    def reset_from_state(self, state):
        ob_dict = self.env.reset_from_state(state)
        return self._flatten_obs(ob_dict)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        # FIXME
        # Could just return the abstract state:
        # self.env.abstract_state(obs_dict)

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

    # def _flatten_abstract_state(self, state_dict):
        # """
        # Each element in abstract state is a scalar.
        # """
        # arr = np.array(list(state_dict.values()))
        # return arr
