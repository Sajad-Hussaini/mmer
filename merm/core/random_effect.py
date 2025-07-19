import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse.linalg import cg

class RandomEffect:
    """
    Random Effect class encapsulates the design and computation of random effects in a mixed model context.
    It constructs the random effects design matrix (Z), computes the conditional mean (μ),
    and manages the covariance structure of the random effects.
    """
    def __init__(self, n: int, m: int, group_col: int, covariates_cols: list[int]):
        self.n = n
        self.m = m
        self.col = group_col
        self.covariates_cols = covariates_cols
        self.q = None
        self.o = None
        self.cov = None
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
        self.ZTZ = self.Z.T @ self.Z
        return self
    
    def compute_mu(self, prec_resid):
        """
        Computes the random effect conditional mean μ.
        takes: 
            1d array (mn,) prec_resid: precision-weighted residuals V⁻¹(y-fx)
        returns:
            1d array (mqo,) μ: conditional mean of the random effects
        """
        return self.kronZ_D_T_matvec(prec_resid)

    def map_mu(self, mu):
        """
        Maps the random effect conditional mean μ to the observation space
        takes:
            1d array (mqo,)
        returns:
            1d array (mn,)
        """
        return self.kronZ_matvec(mu)

    def compute_cov_correction(self, V_op, M_op, n_jobs, backend):
        """
        Computes the correction to the residual covariance matrix φ and
        random effect covariance matrix τ
        by constructing the uncertainty correction matrix T: (m, m)
        and W: (m * q, m * q)

        Uses symmetry of the covariance matrix to reduce computations.
        """
        T = np.zeros((self.m, self.m))
        W = np.zeros((self.m * self.q, self.m * self.q))
        results = Parallel(n_jobs=n_jobs, backend=backend,
                           return_as="generator")(delayed(self.cov_correction_per_response)
                                                  (V_op, M_op, col) for col in range(self.m))

        for col, T_lower_traces, W_lower_blocks in results:
            for i, (trace, W_block) in enumerate(zip(T_lower_traces, W_lower_blocks)):
                row = col + i
                # --- Assemble T ---
                T[col, row] = T[row, col] = trace
                # --- Assemble W ---
                r_slice = slice(row * self.q, (row + 1) * self.q)
                c_slice = slice(col * self.q, (col + 1) * self.q)
                W[r_slice, c_slice] = W_block
                if row != col:
                    W[c_slice, r_slice] = W_block.T

        return T, W
    
    def cov_correction_per_response(self, V_op, M_op, col):
        """
        Computes the element of the uncertainty correction matrix T that is:
            Tᵢⱼ = trace((Zₖᵀ Zₖ) Σᵢⱼ)
        using the random effect conditional covariance
            Σ = D - D (Iₘ ⊗ Z)ᵀ V⁻¹ (Iₘ ⊗ Z) D
        as well as the correction matrix W per response.
        """
        block_size = self.q * self.o
        num_blocks = self.m - col
        lower_sigma = np.zeros((num_blocks * block_size, block_size))
        base_idx = col * block_size
        vec = np.zeros(self.m * block_size)
        for i in range(block_size):
            vec[base_idx + i] = 1.0
            rhs = self.kronZ_D_matvec(vec)
            x_sol, _ = cg(V_op, rhs, M=M_op)
            lower_sigma[:, i] = (self.D_matvec(vec) - self.kronZ_D_T_matvec(x_sol))[col * block_size : ]
            vec[base_idx + i] = 0.0

        T_traces = [self.ZTZ.multiply(lower_sigma[i * block_size:(i + 1) * block_size, :]).sum()
                    for i in range(num_blocks)]
        W_lower_blocks = lower_sigma.reshape(num_blocks, self.q, self.o, self.q, self.o).sum(axis=(2, 4))
        return col, T_traces, W_lower_blocks

    def compute_cov(self, mu, W):
        """
        Compute the random effect covariance matrix τ = (U + W) / o
        """
        mur = mu.reshape((self.m * self.q, self.o))
        U = mur @ mur.T
        np.add(U, W, out=U)
        tau = U / self.o + 1e-6 * np.eye(self.m * self.q)
        return tau

# ====================== Matrix-Vector Operations ======================

    def kronZ_D_matvec(self, x_vec):
        """
        Computes the matrix-vector product W @ x_vec, where W = (Iₘ ⊗ Z) D maps a vector from
        the random effects space (pre-weighted by D) to the observation space.
        takes:
            1d array (mqo,)
        returns:
            1d array (mn,)
        """
        A_k = x_vec.reshape((self.m * self.q, self.o)).T @ self.cov
        B_k = self.Z @ A_k.T.reshape((self.m, self.q * self.o)).T
        return B_k.T.ravel()
    
    def kronZ_D_T_matvec(self, x_vec):
        """
        Computes the matrix-vector product Wᵀ @ x_vec, where Wᵀ = D (Iₘ ⊗ Z)ᵀ maps a vector from
        the observation space to the random effects space (post-weighted by D).
        takes:
            1d array (mn,)
        returns:
            1d array (mqo,)
        """
        A_k = self.Z.T @ x_vec.reshape((self.m, self.n)).T
        B_k = A_k.T.reshape((self.m * self.q, self.o)).T @ self.cov
        return B_k.T.ravel()
    
    def kronZ_T_matvec(self, x_vec):
        """
        Computes the matrix-vector product (Iₘ ⊗ Z)ᵀ @ x_vec maps a vector from
        the observation space back to the random effects space.
        takes:
            1d array (mn,)
        returns:
            1d array (mqo,)
        """
        A_k = self.Z.T @ x_vec.reshape((self.m, self.n)).T
        return A_k.T.ravel()

    def kronZ_matvec(self, x_vec):
        """
        Computes the matrix-vector product (Iₘ ⊗ Z) @ x_vec maps a vector from
        the random effects space to the observation space.
        takes:
            1d array (mqo,)
        returns:
            1d array (mn,)
        """
        A_k = self.Z @ x_vec.reshape((self.m, self.q * self.o)).T
        return A_k.T.ravel()
    
    def D_matvec(self, x_vec):
        """
        Computes the random effect covariance matrix-vector product (τ ⊗ Iₒ) @ x_vec.
        takes:
            1d array (mqo,)
        returns:
            1d array (mqo,)
        """
        Dx =  x_vec.reshape((self.m * self.q, self.o)).T @ self.cov
        return Dx.T.ravel()
    
    def full_cov_matvec(self, x_vec):
        """
        Computes the matrix-vector product full_cov @ x_vec,
            where full_cov = (Iₘ ⊗ Z) D (Iₘ ⊗ Z)ᵀ
        is random effect contribution to the marginal covariance.
        takes:
            1d array (mn,)
        returns:
            1d array (mn,)
        """
        A_k = self.kronZ_T_matvec(x_vec)
        B_k = self.kronZ_D_matvec(A_k)
        return B_k
    
    def cov_to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        """
        std = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(std, std)