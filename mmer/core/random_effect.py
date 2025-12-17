import numpy as np
from scipy import sparse

class RandomEffect:
    """
    Random Effect class for mixed effects models.

    Manages design matrix, conditional mean, and covariance structure for random effects.

    Parameters
    ----------
    n : int
        Number of samples.
    m : int
        Number of outputs.
    group_col : int
        Index of grouping factor's column.
    covariates_cols : list of int or None
        Indices of covariate columns from `X` used as random slopes covariates.
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
        Construct random effects design matrix (Z).

        Parameters
        ----------
        group : np.ndarray
            Grouping factor of shape (n_samples,).
        covariates : np.ndarray | None
            Covariate matrix for random slopes of shape (n_samples, n_covariates) or None.

        Returns
        -------
        Z : np.ndarray
            Random effects design matrix.
        q : int
            Number of random-effects types.
        o : int
            Number of unique levels.
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

    def design_random_effect(self, X: np.ndarray, groups: np.ndarray):
        """
        Construct random effect design matrix and covariance.

        Parameters
        ----------
        X : np.ndarray
            Fixed effect covariates of shape (n_samples, n_features).
        groups : np.ndarray
            Grouping factors of shape (n_samples, n_groups).

        Returns
        -------
        self : RandomEffect
            The updated object.
        """
        slope_covariates = X[:, self.covariates_cols] if self.covariates_cols is not None else None
        self.Z, self.q, self.o = self.design_Z(groups[:, self.col], slope_covariates)
        self.cov = np.eye(self.m * self.q)
        self.ZTZ = self.Z.T @ self.Z
        return self
    
    def _compute_mu(self, prec_resid: np.ndarray):
        """
        Compute random effect conditional mean.
        μₖ = Dₖ (Iₘ ⊗ Zₖ)ᵀ V⁻¹ (y - f(X))

        Parameters
        ----------
        prec_resid : np.ndarray
            Precision-weighted residuals V⁻¹(y - f(X)) of shape (m * n,).

        Returns
        -------
        mu : np.ndarray
            Conditional mean of random effects of shape (m * q * o,).
        """
        return self._kronZ_D_T_matvec(prec_resid)

    def _map_mu(self, mu: np.ndarray):
        """
        Map conditional mean to sample space.

        Parameters
        ----------
        mu : np.ndarray
            Conditional mean of random effects (m * q * o).

        Returns
        -------
        mapped : np.ndarray
            Mapped mean in sample space (m * n).
        """
        return self._kronZ_matvec(mu)

    def _compute_cov(self, mu: np.ndarray, W: np.ndarray):
        """
        Compute random effect covariance matrix.

        Parameters
        ----------
        mu : np.ndarray
            Conditional mean of random effects.
        W : np.ndarray
            Covariance matrix.

        Returns
        -------
        tau : np.ndarray
            Random effect covariance matrix.
        """
        mur = mu.reshape((self.m * self.q, self.o))
        tau = mur @ mur.T
        tau += W
        tau /= self.o
        tau[np.diag_indices_from(tau)] += 1e-5
        return tau

# ====================== Matrix-Vector Operations ======================
    
    def _kronZ_D_matvec(self, x_vec: np.ndarray):
        """
        Matrix-vector product: from random effects to sample space.
        (Iₘ ⊗ Zₖ)D @ x_vec

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector (m * q * o).

        Returns
        -------
        result : np.ndarray
            Output vector (m * n).
        """
        A_k = self._D_matvec(x_vec)
        B_k = self._kronZ_matvec(A_k)
        return B_k

    def _kronZ_D_T_matvec(self, x_vec: np.ndarray):
        """
        Matrix-vector product: from sample to random effects space.
        D (Iₘ ⊗ Zₖᵀ) @ x_vec

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector (m * n).

        Returns
        -------
        result : np.ndarray
            Output vector (m * q * o).
        """
        A_k = self._kronZ_T_matvec(x_vec)
        B_k = self._D_matvec(A_k)
        return B_k
    
    def _kronZ_T_matvec(self, x_vec: np.ndarray):
        """
        Matrix-vector product: from sample to random effects space.
        (Iₘ ⊗ Zₖᵀ) @ x_vec

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector (m * n).

        Returns
        -------
        result : np.ndarray
            Output vector (m * q * o).
        """
        A_k = (x_vec.reshape((self.m, self.n)) @ self.Z).ravel()
        return A_k
    
    def _kronZ_matvec(self, x_vec: np.ndarray):
        """
        Matrix-vector product: from random effects to sample space.
        (Iₘ ⊗ Zₖ) @ x_vec

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector in random effects space (m * q * o).

        Returns
        -------
        result : np.ndarray
            Output vector in sample space (m * n).
        """
        A_k = (self.Z @ x_vec.reshape((self.m, self.q * self.o)).T).T.ravel()
        return A_k
    
    def _D_matvec(self, x_vec: np.ndarray):
        """
        Covariance matrix-vector product.
        D @ x_vec

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector (m * q * o).

        Returns
        -------
        result : np.ndarray
            Output vector (m * q * o).
        """
        Dx = (self.cov @ x_vec.reshape((self.m * self.q, self.o))).ravel()
        return Dx
    
    def _full_cov_matvec(self, x_vec: np.ndarray):
        """
        Compute the random effect covariance matrix-vector product.
        (Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ @ x_vec

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector (m * n,).

        Returns
        -------
        np.ndarray
            Result vector of shape (m * n,).
        """
        A_k = self._kronZ_D_T_matvec(x_vec)
        B_k = self._kronZ_matvec(A_k)
        return B_k
    
    def to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix of shape (m * q, m * q).
        """
        std = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(std, std)