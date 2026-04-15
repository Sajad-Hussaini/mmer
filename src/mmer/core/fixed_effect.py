import numpy as np
from sklearn.model_selection import GroupShuffleSplit


class FixedEffectPipeline:
    """
    Encapsulate validation splitting, fixed-effect fitting, and prediction.

    The mixed-effect estimator can delegate fixed-effect orchestration here,
    keeping the EM loop focused on the mixed-effect part only.
    """

    def __init__(self, model):
        self.model = model
        self.train_idx = None
        self.val_idx = None
        self.has_validation = False

    def configure_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        validation_split: float = 0.0,
        validation_group: int = 0,
        random_state: int = 42,
    ):
        if validation_split > 0:
            main_group = groups[:, validation_group] if groups.ndim > 1 else groups
            splitter = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=random_state)
            self.train_idx, self.val_idx = next(splitter.split(X, y, groups=main_group))
            self.has_validation = True
        else:
            self.train_idx = np.arange(X.shape[0])
            self.val_idx = None
            self.has_validation = False

        return self.train_idx, self.val_idx, self.has_validation

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.has_validation:
            X_train = X[self.train_idx]
            y_train = y[self.train_idx]
            X_val = X[self.val_idx]
            y_val = y[self.val_idx]
            self.model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            self.model.fit(X, y)

        return self

    def fit_and_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        fx = np.asarray(self.model.predict(X))
        if fx.ndim == 1:
            fx = fx[:, None]
        return fx
