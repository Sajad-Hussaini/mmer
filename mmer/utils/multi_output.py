import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class MultiOutputRegressor(BaseEstimator, RegressorMixin):
    """
    A custom multi-output regressor that fits a separate model for each target column.
    """
    def __init__(self, estimators: list):
        """
        Initializes the multi-output regressor with a list of estimators.
        """
        self.estimators = estimators

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits each estimator to its corresponding column in the target array.
        Parameters:
            X : shape (n_samples, n_features)
            y : shape (n_samples, n_outputs)
        """
        if y.ndim == 1:
            raise ValueError("y must be a 2D array with shape (n_samples, n_outputs)")

        if len(self.estimators) != y.shape[1]:
            raise ValueError(
                f"The number of estimators ({len(self.estimators)}) must match "
                f"the number of target columns ({y.shape[1]}).")

        for i, estimator in enumerate(self.estimators):
            estimator.fit(X, y[:, i])

        return self

    def predict(self, X):
        """
        Predicts outputs by aggregating predictions from each individual model.
        Parameters:
            X : shape (n_samples, n_features)
        Returns:
            y : shape (n_samples, n_outputs)
        """
        n_samples = X.shape[0]
        n_outputs = len(self.estimators)
        predictions = np.empty((n_samples, n_outputs))

        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(X)
        
        return predictions