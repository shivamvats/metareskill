from gym.spaces import Box
import numpy as np

from .state import State


# XXX Deprecated
class StateSpace(Box):

    """
    Implements a Box state space on the given state variables.

    Also implements a distance metric on the space.
    """

    def __init__(self, var_names, low, high, distance_fn=None):
        print("This class has been deprecated.")
        super().__init__(np.array(low), np.array(high))
        self._var_names = var_names
        self._distance_fn = distance_fn

    def sample(self):
        return State(self._var_names).from_array(self.sample())

    def contains(self, x):
        if isinstance(x, State):
            return super().contains(x.as_array())
        else:
            return super().contains(x)

    def distance(self, a, b):
        if self._distance_fn:
            return self._distance_fn(a, b)
        else:
            if isinstance(a, State):
                a = a.as_array()
                b = b.as_array()
            W = [1.0]*len(a)
            diff = a - b
            return np.sqrt((W * diff * diff).sum())

    @property
    def vars(self):
        return self._var_names

    def __repr__(self):
        repr = f"Vars: {self._var_names}\n"
        repr += super().__repr__()
        return repr
