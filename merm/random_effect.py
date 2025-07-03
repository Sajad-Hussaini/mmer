import numpy as np
from . import utils

class RandomEffect:
    def __init__(self, group_col, n_obs, n_res, covariates_cols):
        self.col = group_col
        self.n_obs = n_obs
        self.n_res = n_res
        self.covariates_cols = covariates_cols
        self.n_effect = None
        self.n_level = None
        self.cov = None
        self.mu = None
        self.Z = None
        self.ZTZ = None

    def design_re(self, X, groups):
        """
        Constructs the random effect design matrix, number of effect type and levels.
        """
        slope_covariates = X[:, self.covariates_cols] if self.covariates_cols is not None else None
        self.Z, self.n_effect, self.n_level = utils.design_re(groups[:, self.col], slope_covariates)
        self.cov = np.eye(self.n_res * self.n_effect)  # Not the best to initialize covariance matrix here
        return self
    
    def cross_product(self):
        """
        Computes the cross-product of the design matrix.
        """
        self.ZTZ = self.Z.T @ self.Z
        return self
    
    def map_mu(self):
        """
        Maps the conditional mean (mu) to the observation space (I_M ⊗ Z_k)μ.
        returns:
            2d array (n, M)
        """
        B = self.mu.reshape((self.n_effect * self.n_level, self.n_res), order='F')
        return self.Z @ B