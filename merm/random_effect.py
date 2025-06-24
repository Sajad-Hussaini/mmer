import numpy as np
from scipy.sparse.linalg import cg
from joblib import Parallel, delayed, cpu_count
from . import utils

NJOBS = max(1, int(cpu_count() * 0.75))

class RandomEffect:    
    def __init__(self, n_obs, n_res, id, slope_id):
        self.n_obs = n_obs
        self.n_res = n_res
        self.id = id
        self.slope_id = slope_id

    def design_matrix(self, X, groups):
        """
        Constructs the random effect design matrix, number of effect type and levels.
        """
        slope_covariates = X[:, self.slope_id] if self.slope_id is not None else None
        self.Z_matrix, self.n_effect, self.n_level = utils.random_effect_design_matrix(groups[:, self.id], slope_covariates)
        self.cov = np.eye(self.n_res * self.n_effect)  # Initialize covariance matrix for random effects!! may not be the best practice
        return self
    
    def crossproduct(self):
        """
        Computes the cross-product of the design matrix.
        """
        self.Z_crossprod = self.Z_matrix.T @ self.Z_matrix
        return self
    
    def _W_matvec(self, x_vec):
        """
        Computes the matrix-vector product W @ x_vec, where W = (I_M ⊗ Z) D maps a vector from
        the random effects space (pre-weighted by D) to the observation space.
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        x_mat = x_vec.reshape((self.n_res * self.n_effect, self.n_level)).T
        A_k = x_mat @ self.cov
        A_k = A_k.reshape((self.n_level, self.n_res, self.n_effect)).transpose(1, 2, 0).reshape((self.n_res, self.n_effect * self.n_level)).T
        B_k = self.Z_matrix @ A_k
        B_k = B_k.T.ravel()  # (M*n, )
        return B_k

    def _W_T_matvec(self, x_vec):
        """
        Computes the matrix-vector product W^T @ x_vec, where W^T = D (I_M ⊗ Z)^T maps a vector from
        the observation space back to the random effects space (post-weighted by D).
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        x_mat = x_vec.reshape((self.n_res, self.n_obs)).T
        A_k = self.Z_matrix.T @ x_mat
        A_k = A_k.reshape((self.n_effect, self.n_level, self.n_res)).transpose(1, 2, 0).reshape((self.n_level, self.n_res * self.n_effect))
        B_k = A_k @ self.cov
        B_k = B_k.reshape((self.n_level, self.n_res, self.n_effect)).transpose(1, 2, 0).ravel()  # (M*q*o, )
        return B_k
    
    def cov_matvec(self, x_vec):
        """
        Computes the matrix-vector product cov @ x_vec, where cov = (I_M ⊗ Z) D (I_M ⊗ Z)^T is
        random effect contribution to the marginal covariance.
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        A_k = self._W_T_matvec(x_vec)
        A_k = A_k.reshape((self.n_res, self.n_effect * self.n_level)).T
        return self.Z_matrix @ A_k  # (n, M)
    
    def cond_mean(self, V_inv_eps):
        """
        Computes the random effect conditional mean by leveraging the kronecker structure.
        """
        self.mu = self._W_T_matvec(V_inv_eps)
        return self
    
    def map_cond_mean(self):
        """
        Maps the conditional mean back to the observation space.
        """
        mean_2d = self.mu.reshape((self.n_res, self.n_effect * self.n_level)).T
        return self.Z_matrix @ mean_2d
    
    def resid_cov(self, V_op):
        """
        Computes the random effect contribution to the residual covariance matrix.
        Uses symmetry of the covariance matrix to reduce computations.
        """
        cov = np.zeros((self.n_res, self.n_res))
        for row in range(self.n_res):
            for col in range(row, self.n_res):
                sigma_block = self.cond_cov_res_block(V_op, row, col)
                trace = np.sum(sigma_block * self.Z_crossprod)
                cov[col, row] = cov[row, col] = trace
        return cov

    def rand_effect_cov(self, V_op):
        """
        Compute the full random effect covariance matrix.
        """
        M, q, o = self.n_res, self.n_effect, self.n_level
        cov = np.zeros((M * q, M * q))

        # Compute indices for all levels
        m_idx = np.arange(M)[:, None]
        q_idx = np.arange(q)[None, :]
        base_idx = m_idx * q * o + q_idx * o
        
        for j in range(o):
            lvl_indices = (base_idx + j).ravel()
            mu_j = self.mu[lvl_indices]
            sigma_block = self.cond_cov_lvl_block(V_op, lvl_indices)
            cov += np.outer(mu_j, mu_j) + sigma_block
        self.cov = cov / o + 1e-6 * np.eye(M * q)
        return self

    def cond_cov_res_block(self, V_op, row, col):
        """
        Computes the random effect conditional covariance
            Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
        for the response block specified by (row, col).
        """
        M, q, o = self.n_res, self.n_effect, self.n_level
        block_size = q * o

        tau_block = self.cov[row * q : (row + 1) * q, col * q : (col + 1) * q]
        D_block = np.kron(tau_block, np.eye(o))
        sigma_block = np.zeros((block_size, block_size))
        base_idx = col * block_size # Extracts columns in W_matvec

        for i in range(block_size):
            vec = np.zeros(M * block_size)
            vec[base_idx + i] = 1.0
            rhs = self._W_matvec(vec)
            x_sol, _ = cg(V_op, rhs)
            sigma_block[:, i] = self._W_T_matvec(x_sol)[row * block_size : (row + 1) * block_size]

        return D_block - sigma_block
    
    def cond_cov_lvl_block(self, V_op, lvl_indices):
        """
        Computes the random effect conditional covariance
            Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
        for the level block specified by lvl_indices.
        """
        M, q, o = self.n_res, self.n_effect, self.n_level
        block_size = M * q

        D_block = self.cov
        sigma_block = np.zeros((block_size, block_size))

        for i in range(block_size):
            vec = np.zeros(block_size * o)
            vec[lvl_indices[i]] = 1.0
            rhs = self._W_matvec(vec)
            x_sol, _ = cg(V_op, rhs)
            sigma_block[:, i] = self._W_T_matvec(x_sol)[lvl_indices]

        return D_block - sigma_block