import logging
import numpy as np

from autolab_core import RigidTransform
from .robot_skill import RobotSkill
from .reps_skill import REPSSkill
from recovery_skills.graph.abstraction import angle_weighted_euclidean_distance
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import robosuite.utils.transform_utils as T

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class NearestNeighborSkill(RobotSkill):
    """Interpolates between given policies using nearest neighbor regression."""

    def __init__(self, skills, goal_constraint, env_cfg):
        super().__init__(goal_constraint)

        self.skills = skills
        self.goal_constraint = goal_constraint
        self.env_cfg = env_cfg

        self.preconds = None

        self.skill_regression_model = self._train_nearest_neighbor_regressor(skills)

    def precondition_satisfied(self, state):
        pass

    def apply(self, env, obs, context=None, render=False, interpolate=False,
              deterministic=True):

        if hasattr(env, 'num_envs'):
            size = env.num_envs
            is_vec_env = True
        else:
            size = 1
            is_vec_env = False

        if is_vec_env:
            X = self._extract_features(obs)
        else:
            X = self._extract_features([obs])
        X = self.normalizer.transform(X)
        policy_params_mean = self.skill_regression_model.predict(X).flatten()

        # eef_pose = RigidTransform(translation=obs['robot_eef:pose/position'],
                                  # rotation=T.quat2mat(obs['robot_eef:pose/quat']),
                                  # from_frame='franka_tool',
                                  # to_frame='world')
        # local_action = policy_params_mean.reshape(2, -1)
        # global_actions = []
        # for action in local_action:
            # pos = action[:3]
            # axisangle = action[3:6]
            # gripper = action[6]
            # local_action = RigidTransform(translation=pos,
                                           # rotation=T.quat2mat(T.axisangle2quat(axisangle)),
                                          # from_frame='franka_tool',
                                          # to_frame='franka_tool')
            # global_action =  eef_pose * local_action
            # global_actions.append((global_action, gripper))

        # global_params = []
        # for action, gripper in global_actions:
            # params = np.concatenate([action.translation, action.axis_angle,
                                     # [gripper]])
            # global_params.append(params)
        # global_params = np.concatenate(global_params)

        low_level_skill = REPSSkill(self.goal_constraint, self.env_cfg)
        policy_params = {"mean": policy_params_mean,
                         "cov": np.zeros((len(policy_params_mean),
                                          len(policy_params_mean)))}
        low_level_skill.update_policy(policy_params)

        return low_level_skill.apply(env, obs, None, render,
                                     deterministic=True, local_frame=True)

    def train_precondition(self):
        pass

    def _train_nearest_neighbor_regressor(self, skills, n_neighbors=None):
        start_states = np.array([skill.start for skill in skills])

        X_train = self._extract_features(start_states)
        self.normalizer = preprocessing.StandardScaler().fit(X_train)
        X_train = self.normalizer.transform(X_train)

        y_train = []
        for skill in skills:
            actions, _, _ = skill.sample_policy(deterministic=True)
            y_train.append(actions.flatten())
        y_train = np.array(y_train)

        if n_neighbors is None:
            n_neighbors = min(len(skills), 3)

        if n_neighbors:
            knn = KNeighborsRegressor(n_neighbors=n_neighbors,
                                    weights='distance').fit(X_train, y_train)
            return knn

        else:
            # XXX Ignore
            n_neighbors = list(range(2, 7))
            param_grid = {'n_neighbors': n_neighbors,
                        'weights': ['distance'],
                        'metric': ['minkowski'],
                        }
            grid = GridSearchCV(KNeighborsRegressor(),
                                param_grid,
                                scoring='neg_mean_squared_error',
                                refit=True)
            grid.fit(X_train, y_train)
            logger.info(f"  Best params: {grid.best_params_}")
            logger.info(f"  Best score: {grid.best_score_}")
            return grid.best_estimator_

    def _train_linear_regressor(self, skills):
        from sklearn.linear_model import LinearRegression

        start_states = np.array([skill.start for skill in skills])

        X_train = self._extract_features(start_states)
        self.normalizer = preprocessing.StandardScaler().fit(X_train)
        X_train = self.normalizer.transform(X_train)

        y_train = []
        for skill in skills:
            actions, _, _ = skill.sample_policy(deterministic=True)
            y_train.append(actions.flatten())
        y_train = np.array(y_train)
        model = LinearRegression().fit(X_train, y_train)
        return model

    @staticmethod
    def _extract_features(obsx):
        X = []
        for obs in obsx:
            # Vector from eef to handle
            handle_to_eef = (
                obs["robot_eef:pose/position"] - obs["handle:pose/position"]
            )
            x = [handle_to_eef]
            # x = [obs['robot_eef:pose/position'],
                 # obs['handle:pose/position']]
            quat = obs['robot_eef:pose/quat']
            # x.append(quat)
            axisangle = T.quat2axisangle(quat)
            x.append(axisangle)
            for var in [
                        # "door:pose/theta",
                        # "robot_eef:pose/quat",
                        # "robot_eef:gripper/position",
                        # "hinge:pose/theta",
                        # "handle:pose/theta"
                        ]:
                val = obs[var]
                if val.ndim == 0:
                    val = [val]
                x.append(val)
            X.append(np.concatenate(x))
        X = np.array(X)

        return X
