import numpy as np
from . import utils

class RandomEffect:
    def __init__(self, n_obs, n_res, id, covariates_id):
        self.n_obs = n_obs
        self.n_res = n_res
        self.id = id
        self.covariates_id = covariates_id
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
        slope_covariates = X[:, self.covariates_id] if self.covariates_id is not None else None
        self.Z, self.n_effect, self.n_level = utils.design_re(groups[:, self.id], slope_covariates)
        self.cov = np.eye(self.n_res * self.n_effect)  # Not the best to initialize covariance matrix here
        return self
    
    def cross_product(self):
        """
        Computes the cross-product of the design matrix.
        """
        self.ZTZ = self.Z.T @ self.Z
        return self
    
    def map_cond_mean(self):
        """
        Maps the conditional mean back to the observation space.
        """
        B = self.mu.reshape((self.n_effect * self.n_level, self.n_res), order='F')
        return self.Z @ B