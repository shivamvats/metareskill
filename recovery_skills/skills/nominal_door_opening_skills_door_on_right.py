import numpy as np

from autolab_core import RigidTransform
from .robot_skill import RobotSkill
from recovery_skills.utils.transforms import *


class ReachHandleSkill(RobotSkill):
    """Moves to the pre-grasp pose for grasping the handle."""
    def __init__(self, preconds=None):
        self.preconds = preconds

    def precondition_satisfied(self, state, context=None):
        if self.preconds:
            return self.preconds.is_satisfied(state, context)
        else:
            return True

    def termcondition_satisfied(self, state, context=None):
        return False

    def apply(self, env, obs, context=None, render=False, interpolate=False,
              gym_env=True):

        if gym_env:
            obs_dict = env.unflatten_obs(obs)
        else:
            obs_dict = obs
        # middle point of the handle
        ee_quat = obs_dict['robot_eef:pose/quat']
        handle_pos = obs_dict['handle:pose/position']
        handle_theta = obs_dict['handle:pose/theta'][0]
        door_theta = obs_dict['door:pose/theta'][0]
        handle_dims = obs_dict['handle:dims']

        door_normal = door_theta - np.pi/2
        door_anti_normal = door_theta + np.pi/2

        actions = []

        # before pre-grasp
        offset1 = np.array([-0.02, 0.1, 0.0])
        grasp_pos1 = handle_pos + offset1
        # 90deg about x axis
        rot_x = mat_about_x(np.pi/2)
        rot_z = mat_about_z(door_anti_normal)
        target_rot1 = np.matmul(rot_z, rot_x)
        grasp_ori1 = T.quat2axisangle(T.mat2quat(target_rot1))
        action1 = np.concatenate([grasp_pos1, grasp_ori1, [-1.0]])
        actions.append(action1)

        offset2 = np.array([0.0, -0.1, 0.0])
        grasp_pos2 = grasp_pos1 + offset2
        grasp_ori2 = grasp_ori1
        action2 = np.concatenate([grasp_pos2, grasp_ori2, [-1.0]])
        actions.append(action2)

        for action in actions:
            for _ in range(50):
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()
        return obs, rew, done, info


class GraspHandleSkill(RobotSkill):
    """Grasps the handle"""
    def __init__(self, preconds=None):
        self.preconds = preconds

    def precondition_satisfied(self, state, context=None):
        if self.preconds:
            return self.preconds.is_satisfied(state, context)
        else:
            return True

    def termcondition_satisfied(self, state, context=None):
        return False

    def apply(self, env, obs, context=None, render=False, interpolate=False,
              gym_env=True):

        if gym_env:
            obs_dict = env.unflatten_obs(obs)
        else:
            obs_dict = obs
        ee_pos = obs_dict['robot_eef:pose/position']
        ee_quat = obs_dict['robot_eef:pose/quat']
        ee_axisangle = T.quat2axisangle(ee_quat)

        action = np.concatenate([ee_pos, ee_axisangle, [1.0]])
        for _ in range(50):
            obs, rew, done, info = env.step(action)
            if render:
                env.render()

        return obs, rew, done, info


class RotateHandleSkill(RobotSkill):
    """Rotates the handle so as to open it"""
    def __init__(self, preconds=None):
        self.preconds = preconds

    def precondition_satisfied(self, state, context=None):
        if self.preconds:
            return self.preconds.is_satisfied(state, context)
        else:
            return True

    def termcondition_satisfied(self, state, context=None):
        return False

    def apply(self, env, obs, context=None, render=False, interpolate=False,
              gym_env=True):

        if gym_env:
            obs_dict = env.unflatten_obs(obs)
        else:
            obs_dict = obs
        # middle point of the handle
        ee_pos = obs_dict['robot_eef:pose/position']
        ee_quat = obs_dict['robot_eef:pose/quat']
        ee_mat = T.quat2mat(ee_quat)
        handle_pos = obs_dict['handle:pose/position']
        handle_theta = obs_dict['handle:pose/theta'][0]
        door_theta = obs_dict['door:pose/theta'][0]
        handle_dims = obs_dict['handle:dims']
        handle_length = handle_dims[0]
        handle_cor_pos = obs_dict['handle_center_of_rotation:pose/position']

        door_normal = door_theta - np.pi/2
        door_anti_normal = door_theta + np.pi/2

        actions = []

        current_ee_T = RigidTransform(translation=ee_pos,
                              rotation=T.quat2mat(ee_quat))

        target_pos = np.array([handle_cor_pos[0], # + 0.02,
                               handle_cor_pos[1],
                               handle_cor_pos[2] - handle_length
                               ])
        rot_mat = mat_about_y(-np.pi/2)
        target_mat = np.matmul(rot_mat, ee_mat)
        target_rot = T.quat2axisangle(T.mat2quat(target_mat))

        start_T = RigidTransform(translation=ee_pos,
                               rotation=ee_mat)
        target_T = RigidTransform(translation=target_pos,
                                  rotation=target_mat)
        interps_T = start_T.linear_trajectory_to(target_T, 3)

        for interp in interps_T[1:]:
            action = np.concatenate([interp.translation,
                                     interp.axis_angle,
                                     [0.0]])
            rot_y = mat_about_y(- handle_theta)
            actions.append(action)

        for action in actions:
            for _ in range(100):
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()

            # align with handle
            # obs = env.unflatten_obs(obs)
            # ee_pos = obs['robot_eef:pose/position']
            # ee_quat = obs['robot_eef:pose/quat']
            # handle_theta = obs['handle:pose/theta'][0]
            # # align with handle about y axis
            # rot_mat = mat_about_y(-handle_theta)
            # target_mat = np.matmul(rot_mat, ee_mat)
            # target_rot = T.quat2axisangle(T.mat2quat(target_mat))
            # target_pos = ee_pos

            # align_action = np.concatenate([target_pos,
                                            # target_rot,
                                            # [0]])
            # for _ in range(100):
                # obs, rew, done, info = env.step(align_action)
                # if render:
                    # env.render()

        return obs, rew, done, info


class PullHandleSkill(RobotSkill):
    """Pulls the handle to open the door"""
    def __init__(self, preconds=None):
        self.preconds = preconds

    def precondition_satisfied(self, state, context=None):
        if self.preconds:
            return self.preconds.is_satisfied(state, context)
        else:
            return True

    def termcondition_satisfied(self, state, context=None):
        return False

    def apply(self, env, obs, context=None, render=False, interpolate=False,
              gym_env=True):

        if gym_env:
            obs_dict = env.unflatten_obs(obs)
        else:
            obs_dict = obs
        # middle point of the handle
        ee_pos = obs_dict['robot_eef:pose/position']
        ee_quat = obs_dict['robot_eef:pose/quat']
        handle_theta = obs_dict['handle:pose/theta'][0]
        handle_dims = obs_dict['handle:dims']
        handle_length = handle_dims[0]
        door_theta = obs_dict['door:pose/theta'][0]
        door_anti_normal = door_theta + np.pi/2

        door_width = 0.22
        effective_door_width = door_width - handle_length

        actions = []

        # first align with the handle
        #--------
        target_pos = ee_pos
        # required to ensure the latch doesn't close
        target_pos[0] += 0.02
        rot_x = mat_about_x(np.pi/2)
        # rot_y = mat_about_y(-np.pi/2 + (np.pi/2 - handle_theta))
        rot_y = mat_about_y(- handle_theta)
        rot_z = mat_about_z(door_anti_normal)
        target_mat = np.matmul(rot_y, np.matmul(rot_x, rot_z))
        target_rot = T.quat2axisangle(T.mat2quat(target_mat))
        action = np.concatenate([target_pos,
                                 target_rot,
                                 [0]])
        actions.append(action)

        for _ in range(50):
            env.step(action)
            if render:
                env.render()

        rotate_by_angle = np.pi/6
        # pull_distance = 2 * effective_door_width * np.sin(rotate_by_angle/2)
        pull_distance = 0.2

        current_ee_T = RigidTransform(translation=ee_pos,
                              rotation=T.quat2mat(ee_quat))

        target_pos = np.array([ee_pos[0],
                               ee_pos[1] + pull_distance,
                               ee_pos[2]
                               ])
        ee_mat = T.quat2mat(ee_quat)
        rotate_mat = mat_about_z(rotate_by_angle)
        target_mat = np.matmul(rotate_mat, ee_mat)

        target_ee_T = RigidTransform(translation=target_pos,
                                     rotation=target_mat)

        interps_T = current_ee_T.linear_trajectory_to(target_ee_T, 4)

        for interp in interps_T[1:]:
            action = np.concatenate([interp.translation, interp.axis_angle, [0]])
            actions.append(action)

        for action in actions:
            for _ in range(100):
                obs, rew, done, info = env.step(action)
                if render:
                    env.render()

        return obs, rew, done, info
