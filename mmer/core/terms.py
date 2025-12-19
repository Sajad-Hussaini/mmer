import numpy as np

class RandomEffectTerm:
    """
    Represents the learned structure and state of a random effect.
    This object is data-agnostic regarding the number of samples (n).
    It holds the learned covariance matrix (D).
    """
    def __init__(self, group_id: int, covariates_id: list[int] | None, m: int):
        self.group_id = group_id
        self.covariates_id = covariates_id
        self.m = m        
        self.q = 1 + (len(covariates_id) if covariates_id is not None else 0)
        self.cov = np.eye(self.m * self.q)

    def set_cov(self, new_cov: np.ndarray):
        """Update the learned covariance matrix."""
        if new_cov.shape != (self.m * self.q, self.m * self.q):
            raise ValueError(f"Covariance shape mismatch. Expected {(self.m * self.q, self.m * self.q)}, got {new_cov.shape}")
        self.cov = new_cov

class ResidualTerm:
    """
    Represents the residual variance structure.
    Data-agnostic state holder for R.
    """
    def __init__(self, m: int):
        self.m = m
        self.cov = np.eye(m)

    def set_cov(self, new_cov: np.ndarray):
        if new_cov.shape != (self.m, self.m):
            raise ValueError(f"Residual covariance shape mismatch. Expected {(self.m, self.m)}, got {new_cov.shape}")
        self.cov = new_cov
