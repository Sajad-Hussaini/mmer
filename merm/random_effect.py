import numpy as np
from . import utils

class RandomEffect:
    def __init__(self, n_obs, n_res, id, slope_id):
        self.n_obs = n_obs
        self.n_res = n_res
        self.id = id
        self.slope_id = slope_id
        self.n_effect = None
        self.n_level = None
        self.cov = None
        self.mu = None
        self.Z_matrix = None
        self.Z_crossprod = None

    def design_matrix(self, X, groups):
        """
        Constructs the random effect design matrix, number of effect type and levels.
        """
        slope_covariates = X[:, self.slope_id] if self.slope_id is not None else None
        self.Z_matrix, self.n_effect, self.n_level = utils.random_effect_design_matrix(groups[:, self.id], slope_covariates)
        self.cov = np.eye(self.n_res * self.n_effect)  # Not the best to initialize covariance matrix here
        return self
    
    def crossproduct(self):
        """
        Computes the cross-product of the design matrix.
        """
        self.Z_crossprod = self.Z_matrix.T @ self.Z_matrix
        return self
    
    def map_cond_mean(self):
        """
        Maps the conditional mean back to the observation space.
        """
        mu_reshped = self.mu.reshape((self.n_res, self.n_effect * self.n_level)).T
        return self.Z_matrix @ mu_reshped