from autolab_core import RigidTransform as RT
from icecream import ic
import numpy as np
import robosuite.utils.transform_utils as T


def obs_to_rigid_transform(obs):
    trans = obs["robot_eef:pose/position"]
    quat = obs["robot_eef:pose/quat"]
    quat = T.convert_quat(quat, to="wxyz")
    rot = RT.rotation_from_quaternion(quat)
    transform = RT(translation=trans, rotation=rot)
    return transform


def action_to_rigid_transform(action):
    transform = RT(
        translation=action[:3], rotation=RT.rotation_from_axis_angle(action[3:6])
    )
    return transform


def distance_between_quats(q1, q2, rescale=True):
    """
    Corresponds to the angle of rotation required to go from q1 to q2.
        Range: [0, pi/2]

        If `rescale`, then
            Range: [0, 1]
    """
    # square causes numerical issues
    # range: [0, pi]
    # cos_theta = 2*np.dot(q1, q2)**2 - 1

    # no square
    cos_theta = np.abs(np.dot(q1, q2))

    # enforce cos(theta) range to handle numerical errors
    cos_theta_clipped = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta_clipped)
    if rescale:
        theta = theta/(np.pi/2)
    return theta

def mat_about_x(angle):
    mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)]
    ])
    return mat

def mat_about_y(angle):
    mat = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)]
    ])
    return mat

def mat_about_z(angle):
    mat = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return mat
