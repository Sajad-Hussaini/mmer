import numpy as np

class RandomEffectTerm:
    """
    Learned state of a random effect component.
    
    Stores the covariance matrix (D) and configuration for a specific grouping factor.
    This object is data-agnostic and persists across training/inference.

    Parameters
    ----------
    group_id : int
        Index of the grouping column in `groups`.
    covariates_id : list of int or None
        Indices of columns in `X` for random slopes. None implies random intercept only.
    m : int
        Number of output responses.
    """
    def __init__(self, group_id: int, covariates_id: list[int] | None, m: int):
        self.group_id = group_id
        self.covariates_id = covariates_id
        self.m = m        
        self.q = 1 + (len(covariates_id) if covariates_id is not None else 0)
        self.cov = np.eye(self.m * self.q)

    def set_cov(self, new_cov: np.ndarray):
        """
        Update the learned covariance matrix.

        Parameters
        ----------
        new_cov : np.ndarray
            New covariance matrix of shape (m*q, m*q).
        """
        if new_cov.shape != (self.m * self.q, self.m * self.q):
            raise ValueError(f"Covariance shape mismatch. Expected {(self.m * self.q, self.m * self.q)}, got {new_cov.shape}")
        self.cov = new_cov

class ResidualTerm:
    """
    Learned state of the residual error.

    Stores the residual covariance matrix for the multi-response system.

    Parameters
    ----------
    m : int
        Number of output responses.
    """
    def __init__(self, m: int):
        self.m = m
        self.cov = np.eye(m)

    def set_cov(self, new_cov: np.ndarray):
        """
        Update the residual covariance matrix.

        Parameters
        ----------
        new_cov : np.ndarray
            New covariance matrix of shape (m, m).
        """
        if new_cov.shape != (self.m, self.m):
            raise ValueError(f"Residual covariance shape mismatch. Expected {(self.m, self.m)}, got {new_cov.shape}")
        self.cov = new_cov
