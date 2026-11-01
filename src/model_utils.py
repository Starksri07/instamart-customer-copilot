
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample


class RandomOversampledClassifier(ClassifierMixin, BaseEstimator):
    """Randomly oversample minority classes inside each model fit."""

    def __init__(self, estimator, random_state=42):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        target_count = counts.max()

        X_parts = [X]
        y_parts = [y]

        for class_label, count in zip(classes, counts):
            if count < target_count:
                X_class = X[y == class_label]

                X_extra = resample(
                    X_class,
                    replace=True,
                    n_samples=target_count - count,
                    random_state=self.random_state
                )

                y_extra = np.full(
                    target_count - count,
                    class_label
                )

                X_parts.append(X_extra)
                y_parts.append(y_extra)

        X_balanced = np.concatenate(X_parts, axis=0)
        y_balanced = np.concatenate(y_parts, axis=0)

        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(len(y_balanced))

        X_balanced = X_balanced[order]
        y_balanced = y_balanced[order]

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_balanced, y_balanced)

        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
