from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class State(object):

    """
    Represents a high-level state.

    Imlements utility methods for loading from and converting to dict,
    pillar_state and numpy array.

    Works with and without an underlying state space.
    """
    _state_vars: list
    _var_ndims: list
    _state_space: Any = None

    # Internal
    _state_arr: Any = None
    _state_dict: Any = None

    def __getitem__(self, key):
        return self._state_dict[key]

    def __setitem__(self, key, value):
        assert value.shape == self._state_dict[key].shape

    def __contains__(self, key):
        return key in self._state_dict

        self._state_dict[key] = value
        self._state_arr = self._dict_to_array(self._state_dict)

    def from_array(self, array):
        self._state_arr = np.array(array)
        self._state_dict = self.as_ordered_dict()
        return self

    def from_dict(self, state_dict):
        self._state_dict = deepcopy(state_dict)
        return self

    def from_pillar_state(self, pillar_state):
        # self._state_arr = np.array(pillar_state.get_values_as_vec(self.vars))
        self._state_dict = self.as_ordered_dict()
        return self

    def as_array(self, vars=None):
        return self._dict_to_array(self._state_dict, vars)

    def as_ordered_dict(self):
        state_dict = OrderedDict()

        if self._state_dict is not None:
            for var in self.vars:
                state_dict[var] = self._state_dict[var]

        else:
            if self._state_arr is not None:
                idx = 0
                for var, ndim in zip(self.vars, self._var_ndims):
                    if ndim == 1:
                        state_dict[var] = self._state_arr[idx]
                    else:
                        state_dict[var] = self._state_arr[idx: idx + ndim]
                    idx += ndim

            else:
                raise ValueError

        return state_dict

    def is_close(self, b):
        """Checks if two state are close to each other."""
        a = self._state_arr
        b = b.as_array()
        return np.allclose(a, b, atol=1e-3)

    def _dict_to_array(self, state_dict, vars=None):
        flat_state = []
        if vars is None:
            for key, ndim in zip(self.vars, self._var_ndims):
                val = np.array(state_dict[key])
                if val.ndim == 0:
                    val = [val]
                if not len(val) == ndim:
                    val = [np.nan] * ndim
                flat_state.append(val)

        else:
            for key in vars:
                val = np.array(state_dict[key])
                if val.ndim == 0:
                    val = [val]
                flat_state.append(val)

        array = np.concatenate(flat_state)
        return array

    @property
    def vars(self):
        if self._state_space:
            return self._state_space.vars
        else:
            return self._state_vars

    @property
    def ndim(self):
        return sum(self._var_ndims)

    @property
    def ndims(self):
        return self._var_ndims

    @property
    def space(self):
        return self._state_space
