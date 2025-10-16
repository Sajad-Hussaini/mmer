import numpy as np

class MultiOutputRegressor:
    """
    A custom multi-output regressor that fits a separate model for each target column.

    Parameters
    ----------
    estimators : list
        List of estimator objects for each target output.

    Attributes
    ----------
    estimators : list
        The fitted estimators for each output.
    """
    def __init__(self, estimators: list):
        self.estimators = estimators

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit each estimator to its corresponding column in the target array.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input samples.
        y : np.ndarray, shape (n_samples, n_outputs)
            Target values for each output.

        Returns
        -------
        self : MultiOutputRegressor
            Fitted estimator.
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
        Predict outputs by aggregating predictions from each individual model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : np.ndarray, shape (n_samples, n_outputs)
            Predicted values for each output.
        """
        n_samples = X.shape[0]
        n_outputs = len(self.estimators)
        predictions = np.empty((n_samples, n_outputs))

        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(X)
        
        return predictions