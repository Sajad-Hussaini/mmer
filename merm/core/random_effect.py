import numpy as np
from scipy import sparse

class RandomEffect:
    """
    Random Effect class encapsulates the design and computation of random effects in a mixed model context.
    It constructs the random effects design matrix (Z), computes the conditional mean (μ),
    and manages the covariance structure of the random effects.
    """
    __slots__ = ('n', 'm', 'col', 'covariates_cols', 'q', 'o', 'cov', 'Z', 'ZTZ')
    def __init__(self, n: int, m: int, group_col: int, covariates_cols: list[int] | None):
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
    def design_Z(group: np.ndarray, covariates: np.ndarray):
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
        Constructs the random effect design matrix, number of effect type and levels,
        covariance matrix, and symmetric Gram matrix (G = Zᵀ Z).
        """
        slope_covariates = X[:, self.covariates_cols] if self.covariates_cols is not None else None
        self.Z, self.q, self.o = self.design_Z(groups[:, self.col], slope_covariates)
        self.cov = np.eye(self.m * self.q)
        self.ZTZ = self.Z.T @ self.Z
        return self
    
    def compute_mu(self, prec_resid: np.ndarray):
        """
        Computes the random effect conditional mean μ.
        takes: 
            1d array (mn,) prec_resid: precision-weighted residuals V⁻¹(y-fx)
        returns:
            1d array (mqo,) μ: conditional mean of the random effects
        """
        return self.kronZ_D_T_matvec(prec_resid)

    def map_mu(self, mu: np.ndarray):
        """
        Maps the random effect conditional mean μ to the observation space
        takes:
            1d array (mqo,)
        returns:
            1d array (mn,)
        """
        return self.kronZ_matvec(mu)

    def compute_cov(self, mu: np.ndarray, W: np.ndarray):
        """
        Compute the random effect covariance matrix τ = (U + W) / o
        """
        mur = mu.reshape((self.m * self.q, self.o))
        U = mur @ mur.T
        np.add(U, W, out=U)
        tau = U / self.o + 1e-6 * np.eye(self.m * self.q)
        return tau

# ====================== Matrix-Vector Operations ======================

    def kronZ_D_matvec(self, x_vec: np.ndarray):
        """
        Computes the matrix-vector product W @ x_vec, where W = (Iₘ ⊗ Z) D maps a vector from
        the random effects space (pre-weighted by D) to the observation space.
        takes:
            1d array (mqo,)
        returns:
            1d array (mn,)
        """
        A_k = self.cov @ x_vec.reshape((self.m * self.q, self.o))
        B_k = A_k.reshape((self.m, self.q * self.o)) @ self.Z.T
        return B_k.ravel()

    def kronZ_D_T_matvec(self, x_vec: np.ndarray):
        """
        Computes the matrix-vector product Wᵀ @ x_vec, where Wᵀ = D (Iₘ ⊗ Z)ᵀ maps a vector from
        the observation space to the random effects space (post-weighted by D).
        takes:
            1d array (mn,)
        returns:
            1d array (mqo,)
        """
        A_k = x_vec.reshape((self.m, self.n)) @ self.Z
        B_k = self.cov @ A_k.reshape((self.m * self.q, self.o))
        return B_k.ravel()
    
    def kronZ_T_matvec(self, x_vec: np.ndarray):
        """
        Computes the matrix-vector product (Iₘ ⊗ Z)ᵀ @ x_vec maps a vector from
        the observation space back to the random effects space.
        takes:
            1d array (mn,)
        returns:
            1d array (mqo,)
        """
        A_k = x_vec.reshape((self.m, self.n)) @ self.Z
        return A_k.ravel()

    def kronZ_matvec(self, x_vec: np.ndarray):
        """
        Computes the matrix-vector product (Iₘ ⊗ Z) @ x_vec maps a vector from
        the random effects space to the observation space.
        takes:
            1d array (mqo,)
        returns:
            1d array (mn,)
        """
        A_k = x_vec.reshape((self.m, self.q * self.o)) @ self.Z.T
        return A_k.ravel()
    
    def D_matvec(self, x_vec: np.ndarray):
        """
        Computes the random effect covariance matrix-vector product (τ ⊗ Iₒ) @ x_vec.
        takes:
            1d array (mqo,)
        returns:
            1d array (mqo,)
        """
        Dx =  self.cov @ x_vec.reshape((self.m * self.q, self.o))
        return Dx.ravel()
    
    def full_cov_matvec(self, x_vec: np.ndarray):
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
    
    def to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        """
        std = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(std, std)