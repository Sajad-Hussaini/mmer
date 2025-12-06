import numpy as np

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
        if not hasattr(self, "estimators_"):
            self.estimators_ = []

        for estimator, y_indices in self.estimator_groups:
            if not y_indices:
                raise ValueError("Each estimator group must have at least one target index.")

            y_subset = y[:, y_indices]

            # If an estimator handles a single output, ensure y is 1D
            if y_subset.shape[1] == 1:
                y_subset = y_subset.ravel()

            estimator.fit(X, y_subset)
            self.estimators_.append(estimator)

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
        # Determine the total number of output columns from the groups
        all_indices = [idx for _, indices in self.estimator_groups for idx in indices]
        n_outputs = max(all_indices) + 1

        # Initialize an empty array to store all predictions
        y_pred = np.zeros((X.shape[0], n_outputs))

        for estimator, y_indices in self.estimator_groups:
            pred_subset = estimator.predict(X)

            # If prediction is 1D, reshape to 2D for consistent indexing
            if pred_subset.ndim == 1:
                pred_subset = pred_subset.reshape(-1, 1)

            # Place the predictions into the correct columns of the final output array
            y_pred[:, y_indices] = pred_subset

        return y_pred