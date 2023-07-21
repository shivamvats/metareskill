import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class BayesClassifier():
    """Implements a bayes classifier using a positive and a negative
    distribution"""

    def __init__(self, neg_dist, pos_dist, prior):
        self.models = [neg_dist, pos_dist]
        self.prior = prior
        self.classes = [0, 1]

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X) for model in self.models]).T
        probs = np.exp(logprobs)
        result = probs * self.prior
        # returns probability of positive label
        return (result / result.sum(1, keepdims=True)).T[1]

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), 1)]
