"""Scikit-learn model interfaces."""


class SklearnModel:
    """Simple wrapper around a scikit-learn estimator."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        if self.estimator is None:
            raise ValueError("No estimator configured")
        return self.estimator.fit(X, y)

    def predict(self, X):
        if self.estimator is None:
            raise ValueError("No estimator configured")
        return self.estimator.predict(X)
