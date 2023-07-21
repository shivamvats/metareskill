from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances_argmin_min

from recovery_skills.envs import FrankaDoorEnv
from recovery_skills.graph.abstraction import (weighted_euclidean_distance,
                                               angle_weighted_euclidean_distance,
                                               set_euclidean_distance,
                                               )
from recovery_skills.graph import State

class SetGoalConstraint(AbstractGoalConstraint):
    """
    The goal is a set of points.
    """
    def __init__(self, goal_states=None, thresh=0.1,
                 distance_fn=set_euclidean_distance,
                 relevant_cols=None, is_satisfied_fn=None):

        if goal_states is None:
            goal_states = []

        self.goals = goal_states
        self.thresh = thresh

        if self.goals:
            self._goal_arrs = np.array([goal.as_array() for goal in self.goals])
        else:
            self._goal_arrs = None
        self._distance_fn = distance_fn

        # Abstraction
        self.relevant_cols = relevant_cols

        self._is_satisfied_fn = is_satisfied_fn
        # train one class svm
        self._svm = OneClassSVM(nu=0.1, gamma=0.1)
        if not self._goal_arrs is None:
            self._train_one_class_svm()

    def is_satisfied(self, state):
        if self._is_satisfied_fn:
            return self._is_satisfied_fn(self, state)
        else:
            # return self.distance(state) < self.thresh
            X = state.as_array().reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            # ic(f"SVM score: {self._svm.score_samples(X_scaled)}")
            return self._svm.predict(X_scaled)[0] > 0

    def distance(self, state):
        """Distance from a set of points."""

        # if self.relevant_cols is None:
            # self.relevant_cols = np.arange(self._goal_arrs.shape[1])

        if isinstance(state, State):
            A = state.as_array()
        else:
            A = state
        A = A # [self.relevant_cols]
        B = self._goal_arrs #[:, self.relevant_cols]
        dist, _ = self._distance_fn(A, B)
        return dist

    def get_closest_goal(self, state):
        """Compute the closest goal state in the goal set."""

        if self.relevant_cols is None:
            self.relevant_cols = np.arange(self._goal_arrs.shape[1])

        if isinstance(state, State):
            A = state.as_array()
        else:
            A = state
        A = A[self.relevant_cols]
        B = self._goal_arrs[:, self.relevant_cols]
        _, idx = self._distance_fn(A, B)
        return self.goal[idx]

    def add_goal_state(self, state):
        self.goals.append(state)
        if self._goal_arrs is None:
            self._goal_arrs = state.as_array().reshape(1, -1)
        else:
            self._goal_arrs = np.vstack([self._goal_arrs, state.as_array()])

        self._train_one_class_svm()

    @property
    def goal(self):
        """To support interoperability with GoalConstraint"""
        return self.goals

    def _train_one_class_svm(self):
        X = self._goal_arrs
        self._scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)
        self._svm.fit(X_scaled)


class LearntGoalConstraint(AbstractGoalConstraint):
    """
    The goal is learnt as a function of the context.
    """
    def __init__(self, goal_states,
                 distance_fn=angle_weighted_euclidean_distance,
                 relevant_cols=None, is_satisfied_fn=None):

        # Don't need as prediction is done only using context
        # self.start_contexts = [FrankaDoorEnv.state_to_context(state) for state
                               # in start_states]
        self.goals = goal_states
        self.contexts = [FrankaDoorEnv.state_to_context(state) for state
                                in goal_states]
        self._context_arrs = np.array([context.as_array() for context in
                                        self.contexts])

        # if self.start_contexts:
            # self._start_arrs = np.array([context.as_array() for context in
                                         # self.start_contexts])
        # else:
            # self._start_arrs = None

        if self.goals:
            self._goal_arrs = np.array([goal.as_array() for goal in self.goals])
        else:
            self._goal_arrs = None

        self._distance_fn = distance_fn

        # Abstraction
        self.relevant_cols = relevant_cols

        self._is_satisfied_fn = is_satisfied_fn
        # train one class svm
        self._svm = OneClassSVM(nu=0.05, gamma=0.15)
        self._goal_model = LinearRegression()
        if not self._goal_arrs is None:
            # goal prediction
            self._train_goal_predictor()
            # is satisfied prediction
            self._train_one_class_svm()

    def is_satisfied(self, state):
        if self._is_satisfied_fn:
            return self._is_satisfied_fn(self, state)
        else:
            X = state.as_array().reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            # ic(f"SVM score: {self._svm.score_samples(X_scaled)}")
            return self._svm.predict(X_scaled)[0] > 0

    def distance(self, state):
        """Distance from the predicted goal."""
        # if self.relevant_cols is None:
            # self.relevant_cols = np.arange(self._goal_arrs.shape[1])
        # state_arr = state.as_array()
        # goal_arr = self.goal(state).as_array()
        # dist = self._distance_fn(state_arr, goal_arr)

        dist = self._distance_fn(state, self.goal(state))
        return dist

    def add_goal_state(self, state):
        context = FrankaDoorEnv.state_to_context(state)
        self.contexts.append(context)
        self.goals.append(state)

        if self._goal_arrs is None:
            self._goal_arrs = state.as_array().reshape(1, -1)
        else:
            self._goal_arrs = np.vstack([self._goal_arrs, state.as_array()])

        if self._context_arrs is None:
            self._context_arrs = context.as_array().reshape(1, -1)
        else:
            self._context_arrs = np.vstack([self._context_arrs, context.as_array()])

        self._train_one_class_svm()
        self._train_goal_predictor()

    def goal(self, state=None):
        if state:
            # same as start state
            context = FrankaDoorEnv.state_to_context(state).as_array().reshape(1, -1)
            goal_arr = self._goal_model.predict(context)[0]
            goal = State(FrankaDoorEnv.rl_state_vars,
                         FrankaDoorEnv.rl_state_var_ndims).from_array(goal_arr)
        else:
            goal = self.goals
        return goal

    def _train_one_class_svm(self):
        X = self._goal_arrs
        self._scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)
        self._svm.fit(X_scaled)

    def _train_goal_predictor(self):
        X = self._context_arrs
        # FIXME only predict the state variables, not context vars
        y = self._goal_arrs
        # self._scaler = preprocessing.StandardScaler().fit(X)
        # X_scaled = self._scaler.transform(X)
        self._goal_model.fit(X, y)
"""
