import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.linalg import cg
from . import utils

class RandomEffect:
    """
    Random Effect class to handle residual computations in mixed effects models.
    It provides methods to compute covariance matrices, design matrices, and perform matrix-vector operations.
    """
    def __init__(self, n: int, m: int, group_col: int, covariates_cols: list[int]):
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

    def design_rand_effect(self, X: np.ndarray, groups: np.ndarray):
        """
        Constructs the random effect design matrix, number of effect type and levels.
        """
        slope_covariates = X[:, self.covariates_cols] if self.covariates_cols is not None else None
        self.Z, self.q, self.o = utils.design_rand_effect(groups[:, self.col], slope_covariates)
        return self
    
    def prepare_data(self):
        """
        Prepare initial covariance matrix, expected values, and symmetric Gram matrix.
            G = Zᵀ Z
        """
        if self.q is None:
            raise ValueError("Must call design_rand_effect() first")
        
        self.cov = np.eye(self.m * self.q)
        self.mu = np.zeros((self.o, self.m * self.q))
        self.ZTZ = self.Z.T @ self.Z
        return self
    
    def compute_mu(self, prec_resid):
        """
        Computes the random effect conditional mean μ.
            prec_resid: precision-weighted residuals V⁻¹(y-fx)
            μ: 2d array (o, m*q)
        """
        self.mu[...] = self.kronZ_D_T_matvec(prec_resid)
        return self

    def map_mu(self):
        """
        Maps the random effect conditional mean μ to the observation space
        returns:
            2d array (n, M)
        """
        return self.kronZ_matvec(self.mu)
    
    def compute_resid_cov_correction(self, V_op, M_op, n_jobs):
        """
        Computes the random effect contribution to the residual covariance matrix φ
        by constructing the uncertainty correction matrix T: m x m
        Uses symmetry of the covariance matrix to reduce computations.
        """
        T = np.zeros((self.m, self.m))
        use_parallel = self.m > 1 and self.q * self.o > 1
        if use_parallel:
            results = Parallel(n_jobs, backend='loky')(delayed(self.resid_cov_correction_per_response)
                                                       (V_op, M_op, row, col)
                                                       for row in range(self.m) for col in range(row, self.m))
        else:
            results = [self.resid_cov_correction_per_response(V_op, M_op, row, col)
                       for row in range(self.m) for col in range(row, self.m)]
        for row, col, trace in results:
            T[col, row] = T[row, col] = trace
        return T

    def resid_cov_correction_per_response(self, V_op, M_op, row, col):
        """
        Computes the element of the uncertainty correction matrix T that is:
            Tᵢⱼ = trace((Zₖ⁻ᵀ Zₖ) Σᵢⱼ)
        using the random effect conditional covariance
            Σ = D - D (Iₘ ⊗ Z)⁻ᵀ V⁻¹ (Iₘ ⊗ Z) D
        """
        block_size = self.q * self.o

        tau_block = self.cov[row * self.q : (row + 1) * self.q, col * self.q : (col + 1) * self.q]
        D_block = np.kron(tau_block, np.eye(self.o))
        sigma_block = np.zeros((block_size, block_size))

        base_idx = col * block_size # starting basis col for extraction
        vec = np.zeros(self.m * block_size)
        for i in range(block_size):
            vec.fill(0.0)
            vec[base_idx + i] = 1.0
            rhs = self.kronZ_D_matvec(vec)
            x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
            rht = self.kronZ_D_T_matvec(x_sol)
            sigma_block[:, i] = rht.ravel(order='F')[row * block_size : (row + 1) * block_size]
        np.subtract(D_block, sigma_block, out=sigma_block)
        # return row, col, np.sum(sigma_block * self.ZTZ)
        return row, col, self.ZTZ.multiply(sigma_block).sum()  # element wise multiplication and sum
    
    def compute_cov(self, V_op, M_op, n_jobs):
        """
        Compute the random effect covariance matrix τ = (U + W) / o
        """
        # Compute indices for all levels
        m_idx = np.arange(self.m)[:, None]
        q_idx = np.arange(self.q)[None, :]
        base_idx = m_idx * self.q * self.o + q_idx * self.o

        beta = self.mu.reshape((self.o, self.m * self.q), order='F')
        U = beta.T @ beta

        use_parallel = self.o > 1 and self.m * self.q > 1
        if use_parallel:
            results = Parallel(n_jobs, backend='loky')(delayed(self.re_cov_correction_per_level)
                                                            (V_op, M_op, (base_idx + j).ravel()) for j in range(self.o))
        else:
            results = [self.re_cov_correction_per_level(V_op, M_op, (base_idx + j).ravel()) for j in range(self.o)]

        rh_term = np.sum(results, axis=0)
        self.cov[...] = self.cov + (U - rh_term) / self.o + 1e-6 * np.eye(self.m * self.q)
        return self
    
    def re_cov_correction_per_level(self, V_op, M_op, lvl_indices):
        """
        Computes the right hand term of the random effect conditional covariance Σ:
            D (Iₘ ⊗ Z)⁻ᵀ V⁻¹ (Iₘ ⊗ Z) D
        for the level block specified by lvl_indices.
        """
        block_size = self.m * self.q
        rh_term = np.zeros((block_size, block_size))
        vec = np.zeros(block_size * self.o)
        for i in range(block_size):
            vec.fill(0.0)
            vec[lvl_indices[i]] = 1.0
            rhs = self.kronZ_D_matvec(vec)
            x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
            rht = self.kronZ_D_T_matvec(x_sol)
            rh_term[:, i] = rht.ravel(order='F')[lvl_indices]
        return rh_term

# ====================== Matrix-Vector Operations ======================

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
    
    def kronZ_T_matvec(self, x_vec):
        """
        Computes the matrix-vector product (Iₘ ⊗ Z)⁻ᵀ @ x_vec maps a vector from
        the observation space back to the random effects space.
        returns:
            2d array (q*o, m)
        """
        x_mat = x_vec.reshape((self.n, self.m), order='F')
        A_k = self.Z.T @ x_mat
        return A_k

    def kronZ_matvec(self, x_vec):
        """
        Computes the matrix-vector product (Iₘ ⊗ Z) @ x_vec maps a vector from
        the random effects space to the observation space.
        returns:
            2d array (n, m)
        """
        x_mat = x_vec.reshape((self.q * self.o, self.m), order='F')
        A_k = self.Z @ x_mat
        return A_k
    
    def D_matvec(self, x_vec):
        """
        Computes the random effect covariance matrix-vector product (τ ⊗ Iₒ) @ x_vec.
        returns:
            2d array (o, m*q)
        """
        x_mat = x_vec.reshape((self.o, self.m * self.q), order='F')
        Dx =  x_mat @ self.cov
        return Dx
    
    def full_cov_matvec(self, x_vec):
        """
        Computes the matrix-vector product full_cov @ x_vec,
            where full_cov = (Iₘ ⊗ Z) D (Iₘ ⊗ Z)⁻ᵀ
        is random effect contribution to the marginal covariance.
        returns:
            2d array (n, m)
        """
        A_k = self.kronZ_T_matvec(x_vec)
        B_k = self.kronZ_D_matvec(A_k)
        return B_k

# ====================== Matrix-Matrix Operations if needed later ======================

# def cov_R_matmat(x_mat, resid_cov, n, m):
#     """
#     Computes the residual covariance matrix-mat product (φ ⊗ Iₙ) @ x_mat.
#     returns:
#         3d array (n, n, num_cols)
#     """
#     num_cols = x_mat.shape[1]
#     x_tensor = x_mat.reshape((n, m, num_cols), order='F')
#     A_k_tensor = np.einsum('ijk,jl->ilk', x_tensor, resid_cov)
#     return A_k_tensor

# def kronZ_T_matmat(x_mat, rand_effect):
#     """
#     Computes the matrix-matrix product (Iₘ ⊗ Z)⁻ᵀ @ x_mat maps a matrix from
#     the observation space to the random effects space.
#     returns:
#         3d array (q*o, n, num_cols)
#     """
#     num_cols = x_mat.shape[1]
#     x_tensor = x_mat.reshape((rand_effect.n, rand_effect.m, num_cols), order='F')
#     A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z.T, x_tensor)
#     return A_k_tensor

# def kronZ_matmat(x_mat, rand_effect):
#     """
#     Computes the matrix-matrix product (Iₘ ⊗ Z) @ x_mat maps a matrix from
#     the random effects space to the observation space.
#     returns:
#         3d array (n, n, num_cols)
#     """
#     num_cols = x_mat.shape[1]
#     # Reshape input into a 3D tensor (q*o, n, num_cols)
#     x_tensor = x_mat.reshape((rand_effect.q * rand_effect.o, rand_effect.m, num_cols), order='F')
#     # Apply the same Z matrix to each of the num_cols slices
#     # 'ij,jkl->ikl' means: for each l, do matmul of (i,j) by (j,k)
#     A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z, x_tensor)
#     return A_k_tensor

# def cov_D_matmat(x_mat, rand_effect):
#     """
#     Computes the random effect covariance matrix-matrix product (τ ⊗ Iₒ) @ x_mat.
#     returns:
#         3d array (o, m*q, num_cols)
#     """
#     num_cols = x_mat.shape[1]
#     # Reshape input into a 3D tensor (o, m*q, num_cols)
#     x_tensor = x_mat.reshape((rand_effect.o, rand_effect.m * rand_effect.q, num_cols), order='F')
#     # Apply the same cov matrix to each of the num_cols slices
#     # 'ijk,jl->ilk' means: for each k, do matrix multiply of (i,j) by (j,l)
#     Dx_tensor = np.einsum('ijk,jl->ilk', x_tensor, rand_effect.cov)
#     return Dx_tensor

# def kronZ_D_matmat(x_mat, rand_effect):
#     """
#     Computes the matrix-matrix product W @ X, where W = (Iₘ ⊗ Z) D maps a matrix from
#     the random effects space (pre-weighted by D) to the observation space.
#     returns:
#         3d array (n, n, num_cols)
#     """
#     num_cols = x_mat.shape[1]
#     m, q, o = rand_effect.m, rand_effect.q, rand_effect.o
#     # Step 1: Apply D (from cov_D_matmat)
#     x_tensor = x_mat.reshape((o, m * q, num_cols), order='F')
#     A_k_tensor = np.einsum('ijk,jl->ilk', x_tensor, rand_effect.cov) # Shape (o, m*q, num_cols)
#     # Step 2: Apply (Iₘ ⊗ Z) (from kronZ_matmat)
#     # Reshape the intermediate tensor for the final multiplication
#     A_k_reshaped = A_k_tensor.reshape((q * o, m, num_cols), order='F')
#     B_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z, A_k_reshaped)
#     return B_k_tensor

# def kronZ_D_T_matmat(x_mat, rand_effect):
#     """
#     Computes the matrix-matrix product W⁻ᵀ @ X, where W⁻ᵀ = D (Iₘ ⊗ Z)⁻ᵀ maps a matrix from
#     the observation space to the random effects space (post-weighted by D).
#     returns:
#         3d array (o, m*q, num_cols)
#     """
#     num_cols = x_mat.shape[1]
#     m, q, o, n = rand_effect.m, rand_effect.q, rand_effect.o, rand_effect.n
#     x_tensor = x_mat.reshape((n, m , num_cols), order='F')
#     A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z.T, x_tensor)
#     A_k_reshaped = A_k_tensor.reshape((o, m*q, num_cols), order='F')
#     B_k_tensor = np.einsum('ijk,jl->ilk', A_k_reshaped, rand_effect.cov)
#     return B_k_tensor