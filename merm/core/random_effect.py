import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse.linalg import cg
import tempfile
import os

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

    @staticmethod
    def design_Z(group: np.ndarray, covariates: np.ndarray = None):
        """
        Construct random effects design matrix (Z) for a grouping variable.
            Intercept block: one-hot encoding for group membership
            Slope block: covariate encoding for group membership
        Parameters:
            group: (n_samples,) array of group levels.
            covariates: (n_samples, q) array for random slopes (optional).
        Returns:
            Z: Sparse ndarray (n_samples, q * o).
            q: Number of random effects.
            o: Number of unique levels.
        """
        n = group.shape[0]
        levels, level_indices = np.unique(group, return_inverse=True)
        o = len(levels)
        base_rows = np.arange(n)
        # Components for the first block (intercept)
        all_data = [np.ones(n)]
        all_rows = [base_rows]
        all_cols = [level_indices]
        q = 1
        # Components for the other block (slope)
        if covariates is not None:
            q += covariates.shape[1]
            for col in range(covariates.shape[1]):
                col_offset = (col + 1) * o
                
                all_data.append(covariates[:, col])
                all_rows.append(base_rows)
                all_cols.append(level_indices + col_offset)
        final_data = np.concatenate(all_data)
        final_rows = np.concatenate(all_rows)
        final_cols = np.concatenate(all_cols)
        Z = sparse.csr_array((final_data, (final_rows, final_cols)), shape=(n, q * o))
        return Z, q, o

    def design_rand_effect(self, X: np.ndarray, groups: np.ndarray):
        """
        Constructs the random effect design matrix, number of effect type and levels.
        """
        slope_covariates = X[:, self.covariates_cols] if self.covariates_cols is not None else None
        self.Z, self.q, self.o = self.design_Z(groups[:, self.col], slope_covariates)
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

    def compute_cov_correction(self, V_op, M_op, n_jobs, backend='loky'):
        """
        Computes the correction to the residual covariance matrix φ and
        random effect covariance matrix τ by constructing
        the uncertainty correction matrices T: (m, m) and W: (m * q, m * q)

        Uses symmetry of the covariance matrix to reduce computations
        """
        shape_T = (self.m, self.m)
        shape_W = (self.m * self.q, self.m * self.q)
        dtype = 'float64'

        def worker(col):
            col, T_traces, W_blocks = self.cov_correction_per_response(V_op, M_op, col)
            for i, trace in enumerate(T_traces):
                row = col + i
                T_shared[col, row] = T_shared[row, col] = trace
            for i, block in enumerate(W_blocks):
                row = col + i
                r_slice = slice(row * self.q, (row + 1) * self.q)
                c_slice = slice(col * self.q, (col + 1) * self.q)
                W_shared[r_slice, c_slice] = block
                if row != col:
                    W_shared[c_slice, r_slice] = block.T

        if backend == 'threading':
            T_shared = np.zeros(shape_T, dtype=dtype)
            W_shared = np.zeros(shape_W, dtype=dtype)
            Parallel(n_jobs=n_jobs, backend=backend)(delayed(worker)(col) for col in range(self.m))
            return T_shared, W_shared

        t_name, w_name = None, None
        try:
            # Create temporary files and immediately close them to release the lock.
            with tempfile.NamedTemporaryFile(delete=False) as f_T:
                t_name = f_T.name
            with tempfile.NamedTemporaryFile(delete=False) as f_W:
                w_name = f_W.name
            
            # Create the memmap objects. They now hold the file handles.
            T_shared = np.memmap(t_name, dtype=dtype, mode='w+', shape=shape_T)
            W_shared = np.memmap(w_name, dtype=dtype, mode='w+', shape=shape_W)
            
            # Run the parallel computation.
            Parallel(n_jobs=n_jobs, backend=backend)(delayed(worker)(col) for col in range(self.m))
            
            # Create the final in-memory copies from the memmap results.
            T_result = np.array(T_shared)
            W_result = np.array(W_shared)
            
            # Explicitly delete the memmap objects.
            # This triggers garbage collection and closes their file handles.
            del T_shared
            del W_shared
            
        finally:
            # The file handles are released, deletion is done.
            if t_name and os.path.exists(t_name):
                os.remove(t_name)
            if w_name and os.path.exists(w_name):
                os.remove(w_name)

        return T_result, W_result
    
    def cov_correction_per_response(self, V_op, M_op, col):
        """
        Computes the element of the uncertainty correction matrix T that is:
            Tᵢⱼ = trace((Zₖᵀ Zₖ) Σᵢⱼ)
        using the random effect conditional covariance
            Σ = D - D (Iₘ ⊗ Z)ᵀ V⁻¹ (Iₘ ⊗ Z) D
        as well as the correction matrix W per response.
        """
        block_size = self.q * self.o

        tau_block = self.cov[:, col * self.q : (col + 1) * self.q]
        D_block = np.kron(tau_block, np.eye(self.o))
        sigma_block = np.zeros((self.m * block_size, block_size))

        base_idx = col * block_size # starting basis col for extraction
        vec = np.zeros(self.m * block_size)
        for i in range(block_size):
            vec[base_idx + i] = 1.0
            rhs = self.kronZ_D_matvec(vec)
            x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
            rht = self.kronZ_D_T_matvec(x_sol)
            sigma_block[:, i] = rht.ravel(order='F')
            vec[base_idx + i] = 0.0
        np.subtract(D_block, sigma_block, out=sigma_block)

        # Compute T traces (lower triangle only)
        T_traces = [self.ZTZ.multiply(sigma_block[i * block_size:(i + 1) * block_size, :]).sum()
                    for i in range(col, self.m)]
        # Compute W blocks (lower triangle only)
        lower_sigma = sigma_block[col * block_size:, :]
        num_blocks = self.m - col
        W_lower_blocks = lower_sigma.reshape(num_blocks, self.q, self.o, self.q, self.o).sum(axis=(2, 4))
        return col, T_traces, W_lower_blocks

    def compute_cov(self, W):
        """
        Compute the random effect covariance matrix τ = (U + W) / o
        """
        beta = self.mu.reshape((self.m * self.q, self.o)) 
        U = beta @ beta.T 
        tau = (U + W) / self.o + 1e-6 * np.eye(self.m * self.q)
        return tau
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
        Computes the matrix-vector product Wᵀ @ x_vec, where Wᵀ = D (Iₘ ⊗ Z)ᵀ maps a vector from
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
        Computes the matrix-vector product (Iₘ ⊗ Z)ᵀ @ x_vec maps a vector from
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
            where full_cov = (Iₘ ⊗ Z) D (Iₘ ⊗ Z)ᵀ
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
#     Computes the matrix-matrix product (Iₘ ⊗ Z)ᵀ @ x_mat maps a matrix from
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
#     Computes the matrix-matrix product Wᵀ @ X, where Wᵀ = D (Iₘ ⊗ Z)ᵀ maps a matrix from
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