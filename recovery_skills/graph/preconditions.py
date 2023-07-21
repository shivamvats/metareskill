import logging
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import *
from recovery_skills.graph import State
from .goal_constraint import GoalConstraint
from recovery_skills.utils.bayes_classifier import *

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")


class PreconditionClassifier(GoalConstraint):

    """Represents a learnt precondition model."""

    relevant_vars = [
        "robot_eef:pose/position",
        # "robot_eef:pose/quat",
        "robot_eef:gripper/position",
        "hinge:pose/theta",
        "handle:pose/position",
        "handle:pose/theta",
    ]

    def __init__(self, states, rl_states, y, one_class_svm_params=None):
        self.states = states
        self.rl_states = rl_states
        self.y = np.array(y)

        if one_class_svm_params:
            self._one_class_svm_params = one_class_svm_params
        else:
            self._one_class_svm_params = {"nu": 0.05, "gamma": 0.20}

        # for preconditions
        rl_state_arr = self._extract_features(states)
        # rl_state_arr = np.array(
            # [np.concatenate([state[var] for var in self.relevant_vars]) for state in rl_states]
        # )
        # for sampling sim states
        state_arr = np.array([state.as_array() for state in states])
        y_train = self.y

        self.clf = self._train_classifier(rl_state_arr, y_train)
        # self.clf = self._train_classifier(state_arr, y_train)
        # learn on the ground state as we need to sample world states
        self._estimate_init_set(state_arr, y_train)

    def _extract_features(self, obsx, sim=False):
        # simplified = True
        simplified = False
        if simplified:
            X = []
            for obs in obsx:
                # Vector from eef to handle
                handle_to_eef = (
                    obs["robot_eef:pose/position"] - obs["handle:pose/position"]
                )
                x = [handle_to_eef]
                for var in ["robot_eef:gripper/position",
                            "hinge:pose/theta", "handle:pose/theta"]:
                    val = obs[var]
                    if val.ndim == 0:
                        val = [val]
                    x.append(val)
                X.append(np.concatenate(x))
            X = np.array(X)

        else:
            X = []
            for obs in obsx:
                x = []
                for var in self.relevant_vars:
                    val = obs[var]
                    if val.ndim == 0:
                        val = [val]
                    x.append(val)
                x = np.concatenate(x)
                X.append(x)
            X = np.array(X)

        return X

    def is_satisfied(self, state, context=None):
        if isinstance(state, State):
            # X = state.as_array(self.relevant_vars)
            X = state.as_array()
        else:
            X = state
        X = X.reshape(1, -1)
        X = self.normalizer.transform(X)
        pred = self.clf.predict(X)
        return pred[0] > 0

    def sample(self, cls, n=1, env=None, penetration_thresh=0.002):
        """Do rejection sampling of states with inter-body penetration if `env`
        is provided."""

        valid_states = []
        failures = 0
        start_time = time()
        while len(valid_states) < n:
            state_arrs, _ = self.init_set.sample(n)
            states = [cls.create_state().from_array(arr) for arr in state_arrs]

            if env:
                for state in states:
                    env.reset_from_state(state)
                    # for _ in range(50):
                    # env.render()
                    penetrations = [contact.dist for contact in env.sim.data.contact]
                    penetration = np.abs(np.min(penetrations))
                    if penetration < penetration_thresh:
                        logger.debug("Valid state found")
                        valid_states.append(state)
                    else:
                        failures += 1
                        # logger.warning(f"Penetration of {penetration} too high")

            else:
                # no penetration check
                valid_states = states

        if env:
            logger.info("State validity check: ")
            logger.info(f"  Took {time() - start_time} s to generate {n} valid samples")
            logger.info(f"  {failures} failures for {n} valid samples")

        return valid_states[:n]

    def distance(self, state):
        """Returns distance from the decision boundary."""

        # X = state.as_array(self.relevant_vars).reshape(1, -1)
        X = state.as_array().reshape(1, -1)
        X = self.normalizer.transform(X)
        return -self.clf.decision_function(X)[0]

    def goal(self, state, env):
        # TODO ensure same context
        samples = self.sample(env, 100)
        # rl_samples = [env.state_to_rl_state(sample) for sample in samples]
        # X = np.array([state.as_array(self.relevant_vars) for state in rl_samples])
        X = np.array([state.as_array() for state in samples])
        X = self.normalizer.transform(X)
        dists = -self.clf.decision_function(X)
        goal_state = samples[np.argmin(dists, axis=0)]
        self.goal_cache = goal_state
        return goal_state

    def update(self, states, rl_states, y):
        self.states = np.concatenate([self.states, states])
        self.rl_states = np.concatenate([self.rl_states, rl_states])
        self.y = np.concatenate([self.y, y])

        state_arr = np.array([state.as_array() for state in self.states])
        y_train = self.y

        self.clf = self._train_classifier(state_arr, y_train)
        # learn on the ground state as we need to sample world states
        self._estimate_init_set(state_arr, y_train)

    def _train_classifier(self, X_train, y_train):
        # normalize
        try:
            self.normalizer = preprocessing.StandardScaler().fit(X_train)
        except ValueError:
            __import__("ipdb").set_trace()

        X_train = self.normalizer.transform(X_train)

        if np.all(y_train):
            clf = OneClassSVM(**self._one_class_svm_params)
            clf.fit(X_train)

        else:
            param_grid = {
                "C": [0.1, 0.5, 1, 5, 10, 50, 100],
                "gamma": [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                "kernel": ["rbf"],
            }
            grid = GridSearchCV(SVC(), param_grid, refit=True)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
            preds = clf.predict(X_train)
            logger.info("Train classification report:")
            logger.info(classification_report(y_train, preds))

        return clf

    def _estimate_init_set(self, X_train, y_train):
        """Approximates the initiation set with a normal distribution"""
        X_pos = X_train[y_train > 0]
        # self.init_set_mean = np.mean(X_pos, axis=0)
        # self.init_set_cov = np.cov(X_pos, rowvar=False)
        # TODO full covariance matrix
        # logger.warning("Using diag covariance matrix")
        self.init_set = GaussianMixture(
            n_components=1,
            # covariance_type='diag'
            covariance_type="full",
        ).fit(X_pos)
        # fixes bug due to 1 component
        # See https://stackoverflow.com/questions/66634076/gaussian-mixture-model-valueerror-pvals-0-pvals-1-or-pvals-contains-nans
        self.init_set.weights_[0] = 1.0

    def _debug_decision_function(self, cls, n, env):
        """
        Samples states from the init set and checks the classifier.
        """

        states = self.sample(cls, n, env, penetration_thresh=0.05)
        logger.info(f"Sampled {n} feasible state from the init set")
        # rl_states = [env.state_to_rl_state(state) for state in states]

        # for state in states:
        # env.reset_from_state(state)
        # for _ in range(50):
        # env.render()

        preds = np.array([self.is_satisfied(state) for state in states])
        print(preds)
        __import__("ipdb").set_trace()


class BayesPreconditionClassifier(GoalConstraint):
    """
    Learns a bayes classifier using positive and negative labels.

    """
    def __init__(self, obs, y):
        self.obs = np.array(obs)
        self.y = np.array(y)

        self.clf = self._train_classifier(self.obs, self.y)
        # learn a positive dist in the sim state representation
        self.sim_pos_dist = self._learn_sim_dist(self.obs, self.y)

    def is_satisfied(self, obs, context=None, thresh=0.5):
        success_proba = self.prob(obs, context)
        is_sat = success_proba > thresh
        return is_sat

    def prob(self, obs, context=None):
        obs = [obs]
        X = self._preprocess_obs(obs)
        success_proba = self.clf.predict_proba(X)[0]
        return success_proba

    def sample(self, cls, n=1, env=None, penetration_thresh=0.002, scale=1.0):
        """Sample the simulation state"""

        if scale == 'uniform':
            gaussian_mean = self.sim_pos_dist.means_[0]
            gaussian_cov = self.sim_pos_dist.covariances_[0]*scale
        elif scale > 1.0:
            gaussian_mean = self.sim_pos_dist.means_[0]
            gaussian_cov = self.sim_pos_dist.covariances_[0]*scale

        valid_states = []
        failures = 0
        start_time = time()
        while len(valid_states) < n:
            print("Sampling")

            if scale > 1.0:
                state_arrs = np.random.multivariate_normal(gaussian_mean,
                                                           gaussian_cov, size=n)
            else:
                state_arrs, _ = self.sim_pos_dist.sample(n)
            states = [cls.create_state().from_array(arr) for arr in state_arrs]

            if env:
                for i, state in enumerate(states):
                    env.reset_from_state(state)
                    penetrations = [contact.dist for contact in env.sim.data.contact]
                    penetration = np.abs(np.min(penetrations))
                    if penetration < penetration_thresh:
                        logger.debug("Valid state found")
                        valid_states.append(state)
                    else:
                        failures += 1
                        # logger.warning(f"Penetration of {penetration} too high")
                    # print(f"Valid: {len(valid_states)}, failures: {failures}")

            else:
                # no penetration check
                valid_states = states

        if env:
            logger.info("State validity check: ")
            logger.info(f"  Took {time() - start_time} s to generate {n} valid samples")
            logger.info(f"  {failures} failures for {n} valid samples")

        return valid_states[:n]

    def reward(self, obs):
        obs = [obs]
        X = self._preprocess_obs(obs)
        # X = self.normalizer.transform(X)
        logprob = self.pos_dist.score_samples(X)[0]
        # neg_logprob = self.neg_dist.score_samples(X)[0]
        reward = logprob / 10.0

        success = int(self.is_satisfied(obs[0]))
        reward += 10 * success

        # logger.debug(f"    logprob: {logprob}, neg-logprob: {neg_logprob}, succcess: {success}")

        return reward

    def distance(self, obs):
        pass
        # # normaliser
        # obs = [obs]
        # X = self._preprocess_obs(obs)
        # # X = self.normalizer.transform(X)
        # prob = self.clf.predict_proba(X)
        # return  1 - prob

    def goal(self, obs, env):
        # TODO ensure same context
        # samples = self.sample(env, 100)
        # X = [sample.as_array() for sample in samples]
        # __import__('ipdb').set_trace()
        # X = self._preprocess_obs(samples)
        # rews = self.sim_pos_dist.score_samples(X)
        # goal_state = samples[np.argmax(rews, axis=0)]
        goal_state = self.sample(env, n=1)[0]
        return goal_state

    def update(self, obs, y):
        pass

    def _train_classifier(self, obs, y):
        # preprocess observatiosn
        X = self._preprocess_obs(obs)
        # self.normalizer = preprocessing.StandardScaler().fit(X)
        # X = self.normalizer.transform(X)

        X_pos = X[np.where(y)]
        X_neg = X[np.where(np.logical_not(y))]

        self.pos_dist = GaussianMixture(n_components=1, covariance_type="full").fit(
            X_pos
        )
        self.neg_dist = GaussianMixture(n_components=4, covariance_type="full").fit(
            X_neg
        )
        self._fix_gmms()

        logger.info("Negative Dist:")
        logger.info(f" Log likelihood of neg: {self.neg_dist.score(X_neg)}")
        logger.info(f" Log likelihood of pos: {self.neg_dist.score(X_pos)}")

        logger.info("Positive Dist:")
        logger.info(f" Log likelihood of neg: {self.pos_dist.score(X_neg)}")
        logger.info(f" Log likelihood of pos: {self.pos_dist.score(X_pos)}")

        pos_covar = self.pos_dist.covariances_[0]
        pos_std = np.sqrt(np.diag(pos_covar))
        fig, axs = plt.subplots(2)
        axs[0].imshow(pos_covar, cmap="gray")
        axs[0].set_title("Covar matrix")
        axs[1].scatter(np.arange(len(pos_std)), pos_std)
        axs[1].set_title("Std")
        fig.savefig("pos_dist_covar.png")

        fig, axs = plt.subplots(2)
        neg_covar = self.neg_dist.covariances_[0]
        neg_std = np.sqrt(np.diag(neg_covar))
        axs[0].imshow(neg_covar, cmap="gray")
        axs[0].set_title("Covar matrix")
        axs[1].scatter(np.arange(len(neg_std)), neg_std)
        axs[1].set_title("Std")
        fig.savefig("neg_dist_covar.png")

        prior = [
            len(X_neg) / (len(X_neg) + len(X_pos)),
            len(X_pos) / (len(X_neg) + len(X_pos)),
        ]

        logger.info(f"Prior: {prior}")

        clf = BayesClassifier(self.neg_dist, self.pos_dist, prior)

        return clf

    def _learn_sim_dist(self, obs, y):
        # preprocess observatiosn
        X = self._preprocess_obs(obs, sim=True)

        X_pos = X[np.where(y)]
        sim_pos_dist = GaussianMixture(n_components=1, covariance_type="full").fit(
            X_pos
        )
        sim_pos_dist.weights_[0] = 1.0

        logger.info("Positive Dist:")
        logger.info(f" Log likelihood of pos: {sim_pos_dist.score(X_pos)}")

        pos_covar = self.pos_dist.covariances_[0]
        pos_std = np.sqrt(np.diag(pos_covar))
        fig, axs = plt.subplots(2)
        axs[0].imshow(pos_covar, cmap="gray")
        axs[0].set_title("Covar matrix")
        axs[1].scatter(np.arange(len(pos_std)), pos_std)
        axs[1].set_title("Std")
        fig.savefig("sim_pos_dist_covar.png")

        return sim_pos_dist

    def _preprocess_obs(self, obs, **kwargs):
        X_train = self._extract_features(obs, **kwargs)
        return X_train

    def _fix_gmms(self):
        # fixes bug due to 1 component
        # See https://stackoverflow.com/questions/66634076/gaussian-mixture-model-valueerror-pvals-0-pvals-1-or-pvals-contains-nans
        self.pos_dist.weights_[0] = 1.0
        self.neg_dist.weights_[0] = 1.0

    @staticmethod
    def _extract_features(obsx, sim=False):
        if sim:
            relevant_vars = [
                "robot:arm/joints",
                "door:pose/position",
                "door:pose/theta",
                "hinge:pose/theta",
                "handle:pose/position",
                "handle:pose/theta",
            ]
            X = []
            for obs in obsx:
                x = []
                for var in relevant_vars:
                    val = obs[var]
                    if val.ndim == 0:
                        val = [val]
                    x.append(val)
                x = np.concatenate(x)
                X.append(x)
            X = np.array(X)

        else:
            relevant_vars = [
                "robot_eef:pose/position",
                # "robot_eef:pose/quat",
                "robot_eef:gripper/position",
                # "door:pose/position",
                "hinge:pose/theta",
                "handle:pose/position",
                "handle:pose/theta",
                # "handle_center_of_rotation:pose/position",
            ]

            simplified = True
            # simplified = False
            if simplified:
                X = []
                for obs in obsx:
                    # Vector from eef to handle
                    handle_to_eef = (
                        obs["robot_eef:pose/position"] - obs["handle:pose/position"]
                    )
                    x = [handle_to_eef]
                    for var in ["robot_eef:gripper/position",
                                "hinge:pose/theta", "handle:pose/theta"]:
                        val = obs[var]
                        if val.ndim == 0:
                            val = [val]
                        x.append(val)
                    X.append(np.concatenate(x))
                X = np.array(X)

            else:
                X = []
                for obs in obsx:
                    x = []
                    for var in relevant_vars:
                        val = obs[var]
                        if val.ndim == 0:
                            val = [val]
                        x.append(val)
                    x = np.concatenate(x)
                    X.append(x)
                X = np.array(X)

        return X


class DoorOpenStartPrecondition(BayesPreconditionClassifier):

    def __init__(self, obs, y, env_cfg):
        super().__init__(obs, y)
        self.env_cfg = env_cfg
        self.home_pos = np.array(env_cfg['eef_home']['position'])

    def is_satisfied(self, obs, context=None):
        eef_pos = obs['robot_eef:pose/position']
        eef_quat = obs['robot_eef:pose/quat']
        hinge = obs['hinge:pose/theta']

        if hinge < 0.04:
            dist_to_home = np.linalg.norm(np.array(eef_pos) - self.home_pos)
            if dist_to_home < 0.2:
                return True
            else:
                return False
        else:
            return False

    def prob(self, obs, context=None):
        return float(self.is_satisfied(obs, context))

    def _train_classifier(self, obs, y):
        # preprocess observatiosn
        X = self._preprocess_obs(obs)
        # self.normalizer = preprocessing.StandardScaler().fit(X)
        # X = self.normalizer.transform(X)

        X_pos = X[np.where(y)]

        self.pos_dist = GaussianMixture(n_components=1, covariance_type="full").fit(
            X_pos
        )
        self.pos_dist.weights_[0] = 1.0

        logger.info("Positive Dist:")
        logger.info(f" Log likelihood of pos: {self.pos_dist.score(X_pos)}")

        return self.pos_dist


class DoorOpenGoalPrecondition(BayesPreconditionClassifier):

    def __init__(self, obs, y):
        super().__init__(obs, y)

    def is_satisfied(self, obs, context=None):
        hinge_theta = obs['hinge:pose/theta']
        return hinge_theta > 0.3
