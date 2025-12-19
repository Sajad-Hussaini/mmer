import numpy as np
from scipy import sparse
from .terms import RandomEffectTerm, ResidualTerm

class RealizedRandomEffect:
    """
    Represents a Random Effect realized on a specific dataset.
    Binds a RandomEffectTerm (State) with Data (Design Matrix Z).
    """
    def __init__(self, term: RandomEffectTerm, X: np.ndarray, groups: np.ndarray):
        self.term = term
        self.n = X.shape[0]
        self.m = term.m
        
        if term.covariates_id is not None:
             covariates = X[:, term.covariates_id]
        else:
             covariates = None
             
        group_data = groups[:, term.group_id]

        self.Z, self.q, self.o = self.design_Z(group_data, covariates)
        
        if self.q != term.q:
            raise ValueError(f"Term q={term.q} does not match realized q={self.q}")

        self.ZTZ = self.Z.T @ self.Z

    @staticmethod
    def design_Z(group: np.ndarray, covariates: np.ndarray | None):
        """
        Construct random effects design matrix (Z).
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

    def _compute_mu(self, prec_resid: np.ndarray):
        """
        Compute random effect conditional mean.
        μₖ = Dₖ (Iₘ ⊗ Zₖ)ᵀ V⁻¹ (y - f(X))
        """
        return self._kronZ_D_T_matvec(prec_resid)

    def _map_mu(self, mu: np.ndarray):
        """
        Map conditional mean to sample space.
        """
        return self._kronZ_matvec(mu)

    def _compute_next_cov(self, mu: np.ndarray, W: np.ndarray):
        """
        Compute the NEW covariance matrix for the term based on this realization.
        Returns 'tau', does NOT update term in place.
        """
        mur = mu.reshape((self.m * self.q, self.o))
        tau = mur @ mur.T
        tau += W
        tau /= self.o
        tau[np.diag_indices_from(tau)] += 1e-5
        return tau
    
    def to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix of shape (m * q, m * q).
        """
        std = np.sqrt(np.diag(self.term.cov))
        return self.term.cov / np.outer(std, std)

# ====================== Matrix-Vector Operations ======================
    
    def _kronZ_D_matvec(self, x_vec: np.ndarray):
        """
        (Iₘ ⊗ Zₖ)D @ x_vec
        """
        A_k = self._D_matvec(x_vec)
        B_k = self._kronZ_matvec(A_k)
        return B_k

    def _kronZ_D_T_matvec(self, x_vec: np.ndarray):
        """
        D (Iₘ ⊗ Zₖᵀ) @ x_vec
        """
        A_k = self._kronZ_T_matvec(x_vec)
        B_k = self._D_matvec(A_k)
        return B_k
    
    def _kronZ_T_matvec(self, x_vec: np.ndarray):
        """
        (Iₘ ⊗ Zₖᵀ) @ x_vec
        """
        A_k = (x_vec.reshape((self.m, self.n)) @ self.Z).ravel()
        return A_k
    
    def _kronZ_matvec(self, x_vec: np.ndarray):
        """
        (Iₘ ⊗ Zₖ) @ x_vec
        """
        A_k = (self.Z @ x_vec.reshape((self.m, self.q * self.o)).T).T.ravel()
        return A_k
    
    def _D_matvec(self, x_vec: np.ndarray):
        """
        D @ x_vec
        """
        Dx = (self.term.cov @ x_vec.reshape((self.m * self.q, self.o))).ravel()
        return Dx
    
    def _full_cov_matvec(self, x_vec: np.ndarray):
        """
        (Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ @ x_vec
        """
        A_k = self._kronZ_D_T_matvec(x_vec)
        B_k = self._kronZ_matvec(A_k)
        return B_k


class RealizedResidual:
    """
    Realized Residual for a specific dataset size n.
    Wraps ResidualTerm (State) and handles n-specific operations.
    """
    def __init__(self, term: ResidualTerm, n: int):
        self.term = term
        self.n = n
        self.m = term.m

    def _compute_next_cov(self, eps: np.ndarray, T_sum: np.ndarray):
        """
        Compute the NEW covariance matrix for the term based on this realization.
        Returns 'phi', does NOT update term in place.
        
        ɸ = (S + T) / n
        """
        epsr = eps.reshape((self.m, self.n))
        phi = epsr @ epsr.T
        phi += T_sum
        phi /= self.n
        phi[np.diag_indices_from(phi)] += 1e-5
        return phi
    
    def _full_cov_matvec(self, x_vec: np.ndarray):
        """
        Compute the residual covariance matrix-vector product.
        Computes (φ ⊗ Iₙ) @ x_vec.
        """
        # Using self.term.cov (phi)
        return (self.term.cov @ x_vec.reshape((self.m, self.n))).ravel()
    
    def to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix of shape (m, m).
        """
        std = np.sqrt(np.diag(self.term.cov))
        return self.term.cov / np.outer(std, std)