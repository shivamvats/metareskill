from icecream import ic
import numpy as np
import robosuite.utils.transform_utils as T
from sklearn.metrics import pairwise_distances_argmin_min
from .state import State
from recovery_skills.utils.transforms import distance_between_quats


def flatten_state_dict(state_dict):
    """
    Flattens vector-values items.
    """
    flattened_state = {}
    for key, var in state_dict.items():
        key0, key1 = key.split('/')
        if key1 == 'position':
            for i, suffix in enumerate(['x', 'y', 'z']):
                flattened_state[key0 + '/' + suffix] = var[i]
        elif key1 == 'joints':
            for i in range(7):
                suffix = f"j{i}"
                flattened_state[key0 + '/' + suffix] = var[i]
        else:
            flattened_state[key] = var
    return flattened_state


def ee_pose_distance(a, b):
    def get_eef_pose(d):
        return np.array([d['robot_eef:pose/x'], d['robot_eef:pose/y'],
                d['robot_eef:pose/z']])

    a = a.as_ordered_dict()
    b = b.as_ordered_dict()
    a_eef_pose = get_eef_pose(a)
    b_eef_pose = get_eef_pose(b)
    diff = a_eef_pose - b_eef_pose
    return np.sqrt((diff * diff).sum())


def weighted_euclidean_distance(a, b, W=None):
    if isinstance(a, State) and isinstance(b, State):
        a, b = a.as_array(), b.as_array()

    if W is None:
        W = np.ones_like(a)
    dist_vec = W * np.abs(a - b)
    return np.linalg.norm(dist_vec)


def angle_weighted_euclidean_distance(a, b, W=None):
    """Uses a different weight for angles."""
    if isinstance(a, State) and isinstance(b, State):
        assert a.ndim == b.ndim, "States are from different spaces"

        a, b = a.as_ordered_dict(), b.as_ordered_dict()
        w_euclid = 1.0
        w_angle = 0.1 # 1.0
        dist_vec = []
        for key in a.keys():
            a_val, b_val = a[key], b[key]
            if key.endswith('quat'):
                shortest_angle = distance_between_quats(a_val, b_val)
                dist = [w_angle * shortest_angle]
                # ic("quat dist", dist)

            elif key.endswith('theta'):
                # scalar
                # dist = [w_angle * np.abs(a_val[0] - b_val[0])]
                dist = [np.abs(a_val[0] - b_val[0])]

            else:
                dist = w_euclid * np.abs(a_val - b_val)
                # ic("xyz  dist:", dist)

            dist_vec.extend(dist)
    else:
        if W is None:
            W = np.ones_like(a)
        dist_vec = W * np.abs(a - b)

    return np.linalg.norm(dist_vec)


def set_euclidean_distance(x, Y):
    """Computes the distance from a point to the closest point in a set Y"""
    X = x.reshape(1, -1)
    idxs, dists = pairwise_distances_argmin_min(X, Y)
    return dists[0], idxs[0]

# FIXME delete
rms_distance = weighted_euclidean_distance

