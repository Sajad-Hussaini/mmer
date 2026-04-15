import numpy as np
from sklearn.base import clone


class MultiOutputRegressor:
    """
    A multi-output regressor that fits a separate estimator for specified groups/individuals of target variables.

    Parameters
    ----------
    estimator_groups : list of tuples
        A list where each tuple contains an estimator instance and a list of
        integer indices for the target variables (y) it should predict.
        Example: [(MLPRegressor(), [0, 1, 2]), (RandomForestRegressor(), [3, 4])]
    """
    def __init__(self, estimator_groups):
        self.estimator_groups = estimator_groups

    def fit(self, X, y):
        """
        Fit the model to data.

        Fits each estimator to its designated subset of the target variables.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input samples.
        y : np.ndarray, shape (n_samples, n_outputs)
            Target values for each output.

        Returns
        -------
        self : object
            Returns self.
        """
        if not self.estimator_groups:
            raise ValueError("At least one estimator group is required.")

        self.estimators_ = []
        self.output_indices_ = []
        self.n_outputs_ = max(idx for _, indices in self.estimator_groups for idx in indices) + 1

        for estimator, y_indices in self.estimator_groups:
            if len(y_indices) == 0:
                raise ValueError("Each estimator group must have at least one target index.")

            y_subset = y[:, y_indices]
            fitted_estimator = clone(estimator)

            # If an estimator handles a single output, ensure y is 1D
            if y_subset.shape[1] == 1:
                y_subset = y_subset.ravel()

            fitted_estimator.fit(X, y_subset)
            self.estimators_.append(fitted_estimator)
            self.output_indices_.append(tuple(y_indices))

        return self

    def predict(self, X):
        """
        Predict regression target for X.

        The prediction of each sample is an aggregation of the predictions
        from each estimator.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : np.ndarray, shape (n_samples, n_outputs)
            Predicted values for each output.
        """
        if not hasattr(self, "estimators_"):
            raise RuntimeError("This MultiOutputRegressor instance is not fitted yet.")

        y_pred = np.zeros((X.shape[0], self.n_outputs_))

        for estimator, y_indices in zip(self.estimators_, self.output_indices_):
            pred_subset = np.asarray(estimator.predict(X))

            # If prediction is 1D, reshape to 2D for consistent indexing
            if pred_subset.ndim == 1:
                pred_subset = pred_subset.reshape(-1, 1)

            # Place the predictions into the correct columns of the final output array
            y_pred[:, y_indices] = pred_subset

        return y_pred