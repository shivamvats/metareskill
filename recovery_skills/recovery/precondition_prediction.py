import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *


class PreconditionPrediction():
    """Given a set of training points and a test point, predicts whether the
    test point will be solved by a regression model trained on the training
    points."""

    def __init__(self, X_train, X_val, y_val):
        self.normalizer = StandardScaler().fit(X_train)
        X_train = self.normalizer.transform(X_train)

        param_grid = {'nu': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                      'gamma': [0.05, 0.1, 0.15, 0.2, 0.5],
                      'kernel': ['rbf']
                      }
        grid = GridSearchCV(OneClassSVM(),
                            param_grid,
                            scoring='accuracy',
                            refit=True)
        grid.fit(X_train)
        self.svm = Pipeline([
            ('normalize', self.normalizer),
            ('clf', grid.best_estimator_)
        ])
        __import__('ipdb').set_trace()

    def is_satisfied(self, x_test):
        pass

    def scoring_fn(self):
        # TODO
        pass
