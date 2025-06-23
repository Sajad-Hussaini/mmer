import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.base import RegressorMixin, clone
from tqdm import tqdm
from . import utils

class RandomEffect:
    def __init__(self, n_obs, n_res, id, slope_id, cov):
        self.n_obs = n_obs
        self.n_res = n_res
        self.id = id
        self.slope_id = slope_id
        self.cov = cov
    
    def design_matrix(self, X, groups):
        slope_covariates = X[:, self.slope_id] if self.slope_id is not None else None
        Z_matrix, self.n_effect, self.n_level = utils.random_effect_design_matrix(groups[:, self.id], slope_covariates)
        return Z_matrix
    
    def crossproduct_matrix(self, design_matrix):
        return design_matrix.T @ design_matrix
    
    def _W_matvec(self, x_vec, design_matrix):
        """
        Computes the matrix-vector product W_k @ x_vec, where W_k = (I_M ⊗ Z_k) D_k maps a vector from
        the random effects space (pre-weighted by D_k) to the observation space.
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        x_mat = x_vec.reshape((self.n_res * self.n_effect, self.n_level)).T
        A_k = x_mat @ self.cov
        A_k = A_k.reshape((self.n_level, self.n_res, self.n_effect)).transpose(1, 2, 0).reshape((self.n_res, self.n_effect * self.n_level)).T
        B_k = design_matrix @ A_k
        B_k = B_k.T.ravel()  # (M*n, )
        return B_k

    def _W_T_matvec(self, x_vec, design_matrix):
        """
        Computes the matrix-vector product W_k^T @ x_vec, where W_k^T = D_k (I_M ⊗ Z_k)^T maps a vector from
        the observation space back to the random effects space (post-weighted by D_k).
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        x_mat = x_vec.reshape((self.n_res, self.n_obs)).T
        A_k = design_matrix.T @ x_mat
        A_k = A_k.reshape((self.n_effect, self.n_level, self.n_res)).transpose(1, 2, 0).reshape((self.n_level, self.n_res * self.n_effect))
        B_k = A_k @ self.cov
        B_k = B_k.reshape((self.n_level, self.n_res, self.n_effect)).transpose(1, 2, 0).ravel()  # (M*q*o, )
        return B_k
    
    def full_cov_matvec(self, x_vec, design_matrix):
        A_k = self._W_T_matvec(x_vec, design_matrix)
        A_k = A_k.reshape((self.n_res, self.n_effect * self.n_level)).T
        return design_matrix @ A_k


class Residual:
    def __init__(self, n_obs, n_res, cov):
        self.n_obs = n_obs
        self.n_res = n_res
        self.cov = cov

    def full_cov_matvec(self, x_vec):
        x_mat = x_vec.reshape((self.n_res, self.n_obs)).T
        return x_mat @ self.cov