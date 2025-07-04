import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.linalg import cg
from . import utils
from . import linalg_op

class RandomEffect:
    def __init__(self, n, m, group_col, covariates_cols):
        self.n = n
        self.m = m
        self.col = group_col
        self.covariates_cols = covariates_cols
        self.q = None
        self.o = None
        self.cov = None
        self.resid_cov_correction = None
        self.mu = None
        self.Z = None
        self.ZTZ = None

    def design_rand_effect(self, X, groups):
        """
        Constructs the random effect design matrix, number of effect type and levels.
        """
        slope_covariates = X[:, self.covariates_cols] if self.covariates_cols is not None else None
        self.Z, self.q, self.o = utils.design_rand_effect(groups[:, self.col], slope_covariates)
        return self
    
    def design_stats(self):
        """
        Initialize covariance matrices and expected values.
        """
        if self.q is None:
            raise ValueError("Must call design_rand_effect() first")
        
        self.cov = np.eye(self.m * self.q)
        self.resid_cov_correction = np.zeros((self.m, self.m))
        self.mu = np.zeros((self.o, self.m * self.q))
        return self
    
    def cross_product(self):
        """
        Computes the cross-product of the design matrix.
        """
        self.ZTZ = self.Z.T @ self.Z
        return self
    
    def compute_mu(self, prec_resid):
        """
        Computes the random effect conditional mean μ as 2d array (o, m*q).
        prec_resid: precision-weighted residuals V⁻¹(y-fx)
            2d array (n, M)
        """
        x_mat = prec_resid.reshape((self.n, self.m), order='F')
        A_k = self.Z.T @ x_mat
        A_k = A_k.reshape((self.o, self.m * self.q), order='F')
        self.mu[...] = A_k @ self.cov
        return self

    def map_mu(self):
        """
        Maps the random effect conditional mean μ to the observation space (Iₘ ⊗ Z)μ.
        returns:
            2d array (n, M)
        """
        B = self.mu.reshape((self.q * self.o, self.m), order='F')
        return self.Z @ B
    
    def compute_resid_cov_correction(self, V_op, M_op, n_jobs):
        """
        Computes the random effect contribution to the residual covariance matrix φ
        by constructing the uncertainty correction matrix T: m x m
        Uses symmetry of the covariance matrix to reduce computations.
        """
        use_parallel = self.m > 0
        if use_parallel:
            results = Parallel(n_jobs, backend='threading')(delayed(self.resid_cov_correction_per_response)
                                                            (V_op, M_op, row, col)
                                                            for row in range(self.m) for col in range(row, self.m))
        else:
            results = [self.resid_cov_correction_per_response(V_op, M_op, row, col) for row in range(self.m) for col in range(row, self.m)]
        for row, col, trace in results:
            self.resid_cov_correction[col, row] = self.resid_cov_correction[row, col] = trace
        return self

    def resid_cov_correction_per_response(self, V_op, M_op, row, col):
        """
        Computes the element of the uncertainty correction matrix T that is:
            Tᵢⱼ = trace((Zₖ⁻ᵀ Zₖ) Σᵢⱼ)
        using the random effect conditional covariance
            Σ = D - D (Iₘ ⊗ Z)⁻ᵀ V⁻¹ (Iₘ ⊗ Z) D
        """
        m, q, o = self.m, self.q, self.o
        block_size = q * o

        tau_block = self.cov[row * q : (row + 1) * q, col * q : (col + 1) * q]
        D_block = np.kron(tau_block, np.eye(o))
        sigma_block = np.zeros((block_size, block_size))

        base_idx = col * block_size # starting basis col for extraction
        vec = np.zeros(m * block_size)
        for i in range(block_size):
            vec.fill(0.0)
            vec[base_idx + i] = 1.0
            rhs = self.kronZ_D_matvec(vec)
            x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
            rht = self.kronZ_D_T_matvec(x_sol)
            sigma_block[:, i] = rht.ravel(order='F')[row * block_size : (row + 1) * block_size]
        np.subtract(D_block, sigma_block, out=sigma_block)
        return row, col, np.sum(sigma_block * self.ZTZ)
    
    def compute_rand_effect_cov(self, V_op, M_op, n_jobs):
        """
        Compute the random effect covariance matrix τ = (U + W) / o
        """
        M, q, o = self.m, self.q, self.o
        # Compute indices for all levels
        m_idx = np.arange(M)[:, None]
        q_idx = np.arange(q)[None, :]
        base_idx = m_idx * q * o + q_idx * o

        beta = self.mu.reshape((o, M * q), order='F')
        U = beta.T @ beta

        use_parallel = o > 0
        if use_parallel:
            results = Parallel(n_jobs, backend='threading')(delayed(self.re_cov_correction_per_level)
                                                            (V_op, M_op, (base_idx + j).ravel()) for j in range(o))
        else:
            results = [self.re_cov_correction_per_level(V_op, M_op, (base_idx + j).ravel()) for j in range(o)]

        rh_term = np.sum(results, axis=0)
        self.cov[...] = self.cov + (U - rh_term) / o + 1e-6 * np.eye(M * q)
        return self
    
    def re_cov_correction_per_level(self, V_op, M_op, lvl_indices):
        """
        Computes the right hand term of the random effect conditional covariance Σ:
            D (Iₘ ⊗ Z)⁻ᵀ V⁻¹ (Iₘ ⊗ Z) D
        for the level block specified by lvl_indices.
        """
        M, q, o = self.m, self.q, self.o
        block_size = M * q

        rh_term = np.zeros((block_size, block_size))

        vec = np.zeros(block_size * o)
        for i in range(block_size):
            vec.fill(0.0)
            vec[lvl_indices[i]] = 1.0
            rhs = self.kronZ_D_matvec(vec)
            x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
            rht = self.kronZ_D_T_matvec(x_sol)
            rh_term[:, i] = rht.ravel(order='F')[lvl_indices]
        return rh_term

    def kronZ_D_matvec(self, x_vec):
        """
        Computes the matrix-vector product W @ x_vec, where W = (Iₘ ⊗ Z) D maps a vector from
        the random effects space (pre-weighted by D) to the observation space.
        returns:
            2d array (n, m)
        """
        x_mat = x_vec.reshape((self.o, self.m * self.q), order='F')
        A_k = x_mat @ self.cov
        A_k = A_k.reshape((self.q * self.o, self.m), order='F')
        B_k = self.Z @ A_k
        return B_k

    def kronZ_D_T_matvec(self, x_vec):
        """
        Computes the matrix-vector product W⁻ᵀ @ x_vec, where W⁻ᵀ = D (Iₘ ⊗ Z)⁻ᵀ maps a vector from
        the observation space to the random effects space (post-weighted by D).
        returns:
            2d array (o, m*q)
        """
        x_mat = x_vec.reshape((self.n, self.m), order='F')
        A_k = self.Z.T @ x_mat
        A_k = A_k.reshape((self.o, self.m * self.q), order='F')
        B_k = A_k @ self.cov
        return B_k