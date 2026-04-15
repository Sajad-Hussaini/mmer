import numpy as np
from abc import ABC, abstractmethod

from .layout import RandomEffectLayout


def _validate_covariance_shape(cov: np.ndarray, expected_shape: tuple[int, int], label: str):
    if cov.shape != expected_shape:
        raise ValueError(f"{label} shape mismatch. Expected {expected_shape}, got {cov.shape}")


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


class RealizedTermBase(ABC):
    """
    Abstract base class for realized terms (random effects and residuals).
    
    Provides common interface for posterior computation and matrix-vector operations
    on data-specific realizations of learned terms.
    """
    
    def __init__(self, term, n: int, m: int):
        """
        Initialize realized term.
        
        Parameters
        ----------
        term : RandomEffectTerm or ResidualTerm
            Learned state (contains covariance).
        n : int
            Dataset size.
        m : int
            Number of outputs.
        """
        self.term = term
        self.n = n
        self.m = m
    
    @abstractmethod
    def _full_cov_matvec(self, x_vec: np.ndarray) -> np.ndarray:
        """Compute full covariance matrix-vector product."""
        pass
    
    def _compute_next_cov(self, *args, **kwargs):
        """
        Estimate new covariance (EM M-step).
        Subclasses override with specific logic.
        """
        raise NotImplementedError


class RandomEffectTerm:
    """
    Learned state of a random effect component.
    
    Stores the covariance matrix (D) and configuration for a specific grouping factor.
    This object is data-agnostic and persists across training/inference.

    Parameters
    ----------
    group_id : int
        Index of the grouping column in `groups`.
    covariates_id : list of int or None
        Indices of columns in `X` for random slopes. None implies random intercept only.
    m : int
        Number of output responses.
    """
    def __init__(self, group_id: int, covariates_id: list[int] | None, m: int):
        self.group_id = group_id
        self.covariates_id = covariates_id
        self.m = m        
        self.q = 1 + (len(covariates_id) if covariates_id is not None else 0)
        self.cov = np.eye(self.m * self.q)

    def set_cov(self, new_cov: np.ndarray):
        """
        Update the learned covariance matrix.

        Parameters
        ----------
        new_cov : np.ndarray
            New covariance matrix of shape (m*q, m*q).
        """
        _validate_covariance_shape(new_cov, (self.m * self.q, self.m * self.q), "Covariance")
        self.cov = new_cov
    
    def to_corr(self, cov: np.ndarray = None) -> np.ndarray:
        """
        Convert a covariance matrix to a correlation matrix.
        If no matrix is provided, use self.cov.
        """
        if cov is None:
            cov = self.cov
        return _cov_to_corr(cov)

    def marginal_cov(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the marginal covariance matrix (m x m) for a single sample
        given its random effect covariate vector z.

        Parameters
        ----------
        z : np.ndarray
            Covariate vector for the sample (shape: (q,)).
            for example, for random intercept + slope, z = [1, x_slope].

        Returns
        -------
        cov : np.ndarray
            Marginal covariance matrix in observation space (shape: (m, m)).
        """
        z = np.asarray(z)
        Im_Z = np.kron(np.eye(self.m), z)
        return Im_Z @ self.cov @ Im_Z.T


class ResidualTerm:
    """
    Learned state of the residual error.

    Stores the residual covariance matrix for the multi-response system.

    Parameters
    ----------
    m : int
        Number of output responses.
    """
    def __init__(self, m: int):
        self.m = m
        self.cov = np.eye(m)

    def set_cov(self, new_cov: np.ndarray):
        """
        Update the residual covariance matrix.

        Parameters
        ----------
        new_cov : np.ndarray
            New covariance matrix of shape (m, m).
        """
        _validate_covariance_shape(new_cov, (self.m, self.m), "Residual covariance")
        self.cov = new_cov
    
    def to_corr(self, cov: np.ndarray = None) -> np.ndarray:
        """
        Convert a covariance matrix to a correlation matrix.
        If no matrix is provided, use self.cov.
        """
        if cov is None:
            cov = self.cov
        return _cov_to_corr(cov)


class RealizedRandomEffect(RealizedTermBase):
    """
    Transient realization of a random effect for a specific dataset Z.
    
    Binds a learned `RandomEffectTerm` (state) to a specific design matrix Z constructed 
    from data X. Used for efficient matrix-vector products in the solver.

    Parameters
    ----------
    term : RandomEffectTerm
        The learned random effect state (e.g., covariance).
    X : np.ndarray
        Fixed effect covariates of shape (n, p).
    groups : np.ndarray
        Grouping factors of shape (n, k).
    """
    def __init__(self, term: "RandomEffectTerm", X: np.ndarray, groups: np.ndarray):
        n = X.shape[0]
        m = term.m
        super().__init__(term, n, m)
        
        if term.covariates_id is not None:
             covariates = X[:, term.covariates_id]
        else:
             covariates = None
             
        group_data = groups[:, term.group_id]

        self.layout = self.design_layout(group_data, covariates)
        self.q = self.layout.q
        self.o = self.layout.o
        
        if self.q != term.q:
            raise ValueError(f"Term q={term.q} does not match realized q={self.q}")

    @staticmethod
    def design_layout(group: np.ndarray, covariates: np.ndarray | None):
        """
        Construct an exact grouped layout for the random effects design.

        Returns
        -------
        layout : RandomEffectLayout
            Group-aware layout with exact matvecs and Gram summaries.
        """
        return RandomEffectLayout(group, covariates)

    @staticmethod
    def design_Z(group: np.ndarray, covariates: np.ndarray | None):
        """
        Compatibility wrapper for older call sites.

        Returns the same exact layout object as design_layout.
        """
        return RealizedRandomEffect.design_layout(group, covariates)

    def _compute_mu(self, prec_resid: np.ndarray):
        """
        Compute posterior mean of random effects.
        μ = D (I_m ⊗ Z^T) V^{-1} r
        """
        return self._kronZ_D_T_matvec(prec_resid)

    def _map_mu(self, mu: np.ndarray):
        """
        Map posterior mean back to observation space.
        y_re = (I_m ⊗ Z) μ
        """
        return self._kronZ_matvec(mu)

    def _compute_next_cov(self, mu: np.ndarray, W: np.ndarray):
        """
        Estimate new covariance D (EM M-step).
        D_new = (μ μ^T + W) / o
        """
        mur = mu.reshape((self.m * self.q, self.o))
        tau = mur @ mur.T
        tau += W
        tau /= self.o
        tau[np.diag_indices_from(tau)] += 1e-5
        return tau

# ====================== Matrix-Vector Operations ======================
    
    def _kronZ_D_matvec(self, x_vec: np.ndarray):
        """(I_m ⊗ Z) D @ x"""
        A_k = self._D_matvec(x_vec)
        B_k = self._kronZ_matvec(A_k)
        return B_k

    def _kronZ_D_T_matvec(self, x_vec: np.ndarray):
        """D (I_m ⊗ Z^T) @ x"""
        A_k = self._kronZ_T_matvec(x_vec)
        B_k = self._D_matvec(A_k)
        return B_k
    
    def _kronZ_T_matvec(self, x_vec: np.ndarray):
        """(I_m ⊗ Z^T) @ x"""
        return self.layout.apply_transpose(x_vec)
    
    def _kronZ_matvec(self, x_vec: np.ndarray):
        """(I_m ⊗ Z) @ x"""
        return self.layout.apply(x_vec)
    
    def _D_matvec(self, x_vec: np.ndarray):
        """D @ x"""
        Dx = (self.term.cov @ x_vec.reshape((self.m * self.q, self.o))).ravel()
        return Dx
    
    def _full_cov_matvec(self, x_vec: np.ndarray):
        """(I_m ⊗ Z) D (I_m ⊗ Z^T) @ x"""
        A_k = self._kronZ_D_T_matvec(x_vec)
        B_k = self._kronZ_matvec(A_k)
        return B_k


class RealizedResidual(RealizedTermBase):
    """
    Transient realization of residuals for a specific dataset size n.
    
    Parameters
    ----------
    term : ResidualTerm
        The learned residual state (e.g., covariance).
    n : int
        Dataset size.
    """
    def __init__(self, term: "ResidualTerm", n: int):
        super().__init__(term, n, term.m)

    def _compute_next_cov(self, eps: np.ndarray, T_sum: np.ndarray):
        """
        Estimate new residual covariance (EM M-step).
        (ε ε^T + T) / n
        """
        epsr = eps.reshape((self.m, self.n))
        phi = epsr @ epsr.T
        phi += T_sum
        phi /= self.n
        phi[np.diag_indices_from(phi)] += 1e-5
        return phi
    
    def _full_cov_matvec(self, x_vec: np.ndarray):
        """
        Compute (R ⊗ I_n) @ x.
        """
        # Using self.term.cov (phi)
        return (self.term.cov @ x_vec.reshape((self.m, self.n))).ravel()
