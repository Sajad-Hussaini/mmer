import numpy as np
from functools import cached_property
from scipy import sparse
from scipy.linalg import cholesky, LinAlgError
from abc import ABC, abstractmethod


def _make_pd(mat: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest positive-definite matrix.

    Clips all eigenvalues to a scale-adaptive floor ``max(min_eig, min_eig * λ_max)``
    and reconstructs via V diag(clipped_λ) V^T.

    Callers are expected to have already symmetrised *mat* (e.g. via tril-copy);
    the interior symmetrisation step is therefore omitted here.
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    floor = max(min_eig, min_eig * float(eigvals[-1]))  # eigvals sorted ascending
    np.maximum(eigvals, floor, out=eigvals)  # in-place clip
    eigvecs *= eigvals  # in-place column-scale (sqrt-factor)
    return eigvecs @ eigvecs.T  # single matmul, no extra broadcast


def _ensure_pd(mat: np.ndarray) -> np.ndarray:
    """
    Cheaply check positive-definiteness via Cholesky; project only on failure.

    The happy path (already PD) costs one Cholesky factorisation and returns
    the original matrix unchanged.  The fallback path (not PD) applies the
    nearest-PD projection.

    Parameters
    ----------
    mat : np.ndarray
        Square symmetric matrix.

    Returns
    -------
    mat : np.ndarray
        The original matrix if PD, otherwise the nearest-PD projection.
    """
    try:
        cholesky(mat, lower=True, check_finite=False)
        return mat
    except LinAlgError:
        return _make_pd(mat)


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
        if new_cov.shape != (self.m * self.q, self.m * self.q):
            raise ValueError(
                f"Covariance shape mismatch. Expected {(self.m * self.q, self.m * self.q)}, got {new_cov.shape}"
            )
        self.cov = new_cov

    def marginal_cov(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the marginal covariance matrix (m x m) for a single observation
        given its random effect design matrix row (covariate vector) z.

        Parameters
        ----------
        z : np.ndarray
            Row of the random effects design matrix for a single observation
            (1D array, shape: (q,)). Since it applies to a single observation
            (and thus a single group level), it reduces to a covariate vector
            containing the intercept (1.0) and any random slopes.
            For example, for random intercept + one slope, z = [1.0, x_slope].

        Returns
        -------
        cov : np.ndarray
            Marginal covariance matrix in observation space (shape: (m, m)).
        """
        z = np.asarray(z)
        if z.ndim != 1:
            raise ValueError(
                f"Covariate vector z must be 1D for a single observation, got shape {z.shape}"
            )
        # Math: (I_m ⊗ z) D (I_m ⊗ z)^T  where D is (m*q, m*q) block matrix.
        # Reshape D to (m, q, m, q) then contract with z on both q-axes:
        # result[i,j] = z^T D[i,j] z  for all (i,j) pairs — no Kronecker product needed.
        D_blocks = self.cov.reshape(self.m, self.q, self.m, self.q)
        return np.einsum("p,iqjr,r->ij", z, D_blocks, z)


class ResidualTerm:
    """
    Learned state of the residual component.

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
        if new_cov.shape != (self.m, self.m):
            raise ValueError(
                f"Residual covariance shape mismatch. Expected {(self.m, self.m)}, got {new_cov.shape}"
            )
        self.cov = new_cov


class RealizedTermBase(ABC):
    """
    Abstract base class for realized terms (random effects and residuals).

    Provides common interface for posterior computation and matrix-vector operations
    on data-specific realizations of learned terms.
    """

    def __init__(self, term, n: int):
        """
        Initialize realized term.

        Parameters
        ----------
        term : RandomEffectTerm or ResidualTerm
            Learned state (contains covariance).
        n : int
            Dataset size.
        """
        self.term = term
        self.n = n
        self.m = term.m

    @abstractmethod
    def _full_cov_matvec(self, x_vec: np.ndarray) -> np.ndarray:
        """Compute full covariance matrix-vector product."""
        pass


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

    def __init__(self, term: RandomEffectTerm, X: np.ndarray, groups: np.ndarray):
        n = X.shape[0]
        super().__init__(term, n)

        if term.covariates_id is not None:
            covariates = X[:, term.covariates_id]
        else:
            covariates = None

        group_data = groups[:, term.group_id]

        self.Z, self.q, self.o = self.design_Z(group_data, covariates)

        if self.q != term.q:
            raise ValueError(f"Term q={term.q} does not match realized q={self.q}")

        self.ZTZ = self.Z.T @ self.Z
        self.ZTZ_diag = self.ZTZ.diagonal().copy()

    @cached_property
    def ZTZ_dense(self) -> np.ndarray:
        """Dense version of Z^T Z, computed lazily and cached for DE correction."""
        return self.ZTZ.toarray()

    @staticmethod
    def design_Z(group: np.ndarray, covariates: np.ndarray | None):
        """
        Construct sparse random effects design matrix Z.

        Returns
        -------
        Z : scipy.sparse.csr_array
            Design matrix of shape (n, q*o).
        q : int
            Number of random parameters per group (intercept + slopes).
        o : int
            Number of unique levels in the grouping factor.
        """
        n = group.shape[0]
        levels, level_indices = np.unique(group, return_inverse=True)
        o = len(levels)
        q = 1 if covariates is None else 1 + covariates.shape[1]

        # Pre-allocate COO arrays for the full sparse matrix in one pass.
        # This avoids building Python lists and calling np.concatenate.
        total_nnz = n * q
        coo_data = np.empty(total_nnz)
        coo_rows = np.empty(total_nnz, dtype=np.intp)
        coo_cols = np.empty(total_nnz, dtype=np.intp)

        base_rows = np.arange(n, dtype=np.intp)
        # Intercept block
        coo_data[:n] = 1.0
        coo_rows[:n] = base_rows
        coo_cols[:n] = level_indices

        if covariates is not None:
            for col in range(covariates.shape[1]):
                sl = slice((col + 1) * n, (col + 2) * n)
                coo_data[sl] = covariates[:, col]
                coo_rows[sl] = base_rows
                coo_cols[sl] = level_indices + (col + 1) * o

        Z = sparse.csr_array((coo_data, (coo_rows, coo_cols)), shape=(n, q * o))
        return Z, q, o

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
        Estimate new covariance D (EM M-step): D_new = (μ μ^T + W) / o.

        Positive-definiteness is enforced lazily: Cholesky is tried first
        (free if already PD); nearest-PD projection is applied only on failure,
        avoiding accumulated-jitter drift over many iterations.
        """
        mur = mu.reshape((self.m * self.q, self.o))
        tau = mur @ mur.T  # symmetric by construction in exact arithmetic
        tau += W  # W is symmetric
        tau /= self.o
        # In-place symmetrisation: copy upper triangle → lower triangle.
        # Uses index arrays of size n*(n-1)/2, avoids a full (n,n) temp matrix.
        _i, _j = np.tril_indices(tau.shape[0], -1)
        tau[_i, _j] = tau[_j, _i]
        return _ensure_pd(tau)

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
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((self.m, self.n, K))
            out = np.empty((self.m, self.q * self.o, K))
            for i in range(self.m):
                out[i] = self.Z.T @ xr[i]
            return out.reshape(self.m * self.q * self.o, K)
        else:
            A_k = (x_vec.reshape((self.m, self.n)) @ self.Z).ravel()
        return A_k

    def _kronZ_matvec(self, x_vec: np.ndarray):
        """(I_m ⊗ Z) @ x"""
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((self.m, self.q * self.o, K))
            out = np.empty((self.m, self.n, K))
            for i in range(self.m):
                out[i] = self.Z @ xr[i]
            return out.reshape(self.m * self.n, K)
        else:
            A_k = (self.Z @ x_vec.reshape((self.m, self.q * self.o)).T).T.ravel()
        return A_k

    def _D_matvec(self, x_vec: np.ndarray):
        """D @ x"""
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((self.m * self.q, self.o, K))
            xr_flat = xr.reshape((self.m * self.q, self.o * K))
            Dx = (
                (self.term.cov @ xr_flat)
                .reshape((self.m * self.q, self.o, K))
                .reshape(self.m * self.q * self.o, K)
            )
        else:
            Dx = (self.term.cov @ x_vec.reshape((self.m * self.q, self.o))).ravel()
        return Dx

    def _full_cov_matvec(self, x_vec: np.ndarray, out: np.ndarray | None = None):
        """(I_m ⊗ Z) D (I_m ⊗ Z^T) @ x"""
        A_k = self._kronZ_D_T_matvec(x_vec)
        B_k = self._kronZ_matvec(A_k)
        if out is not None:
            out += B_k
            return out
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
        super().__init__(term, n)

    def _compute_next_cov(self, eps: np.ndarray, T_sum: np.ndarray):
        """
        Estimate new residual covariance (EM M-step): (ε ε^T + T) / n.

        Positive-definiteness is enforced lazily: Cholesky is tried first
        (free if already PD); nearest-PD projection is applied only on failure,
        avoiding accumulated-jitter drift over many iterations.
        """
        epsr = eps.reshape((self.m, self.n))
        phi = epsr @ epsr.T
        phi += T_sum
        phi /= self.n
        # In-place symmetrisation: copy upper triangle → lower triangle.
        _i, _j = np.tril_indices(phi.shape[0], -1)
        phi[_i, _j] = phi[_j, _i]
        return _ensure_pd(phi)

    def _full_cov_matvec(self, x_vec: np.ndarray, out: np.ndarray | None = None):
        """
        Compute (R ⊗ I_n) @ x.
        """
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((self.m, self.n * K))
            ans = (
                (self.term.cov @ xr)
                .reshape((self.m, self.n, K))
                .reshape(self.m * self.n, K)
            )
        else:
            ans = (self.term.cov @ x_vec.reshape((self.m, self.n))).ravel()

        if out is not None:
            out += ans
            return out
        return ans
