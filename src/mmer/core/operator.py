import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import cho_solve
from .terms import RealizedRandomEffect, RealizedResidual


class VLinearOperator(LinearOperator):
    """
    Linear Operator for the marginal covariance matrix V.

    Represents the matrix-vector product with:
    V = S (I_m \\otimes Z_k) D_k (I_m \\otimes Z_k)^T + R

    Wraps 'RealizedRandomEffect' and 'RealizedResidual' objects to compute
    V @ x efficiently without forming the dense matrix V.

    Parameters
    ----------
    random_effects : tuple of RealizedRandomEffect
        Realized random effect components.
    realized_residual : RealizedResidual
        Realized residual component.
    """

    def __init__(
        self,
        random_effects: tuple[RealizedRandomEffect, ...],
        realized_residual: RealizedResidual,
    ):
        self.random_effects = random_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m
        super().__init__(dtype=np.float64, shape=(self.m * self.n, self.m * self.n))
        
        # Pre-allocate workspaces to avoid allocating arrays at every CG iteration
        self._buf_m_n = np.empty(self.m * self.n)
        self._buf_A_mqo_list = [np.empty(re.m * re.q * re.o) for re in self.random_effects]
        self._buf_B_mqo_list = [np.empty(re.m * re.q * re.o) for re in self.random_effects]
        self._buf_mat_m_n = None
        self._buf_mat_A_mqo_list = None
        self._buf_mat_B_mqo_list = None

    def _matvec(self, x_vec: np.ndarray):
        """Compute V @ x_vec."""
        # Residual part: (R \otimes I_n) x
        Vx = self.realized_residual._full_cov_matvec(x_vec)

        # Random Effects parts
        for i, re in enumerate(self.random_effects):
            re._full_cov_matvec(
                x_vec,
                out=Vx,
                buf_A=self._buf_A_mqo_list[i],
                buf_B=self._buf_B_mqo_list[i],
                buf_C=self._buf_m_n,
            )
        return Vx

    def _matmat(self, x_mat: np.ndarray):
        """Compute V @ x_mat."""
        K = x_mat.shape[1]
        if self._buf_mat_m_n is None or self._buf_mat_m_n.shape[1] != K:
            self._buf_mat_m_n = np.empty((self.m * self.n, K))
            self._buf_mat_A_mqo_list = [np.empty((re.m * re.q * re.o, K)) for re in self.random_effects]
            self._buf_mat_B_mqo_list = [np.empty((re.m * re.q * re.o, K)) for re in self.random_effects]

        # Residual part
        Vx = self.realized_residual._full_cov_matvec(x_mat)

        # Random Effects parts
        for i, re in enumerate(self.random_effects):
            re._full_cov_matvec(
                x_mat,
                out=Vx,
                buf_A=self._buf_mat_A_mqo_list[i],
                buf_B=self._buf_mat_B_mqo_list[i],
                buf_C=self._buf_mat_m_n,
            )
        return Vx

    def _adjoint(self):
        return self

    def __reduce__(self):
        return (self.__class__, (self.random_effects, self.realized_residual))


class ResidualPreconditioner(LinearOperator):
    """
    Preconditioner based on the Residual covariance (R).

    Computes M^{-1} @ x, where M approximation is R.
    P^{-1} = R^{-1} = R_{cov}^{-1} \\otimes I_n.
    """

    def __init__(
        self,
        resid_cov_inv: np.ndarray | None,
        resid_cov_chol: tuple | None,
        n: int,
        m: int,
    ):
        self.cov_inv = resid_cov_inv
        self.cov_chol = resid_cov_chol
        self.n = n
        self.m = m
        super().__init__(dtype=np.float64, shape=(self.m * self.n, self.m * self.n))

    def _matvec(self, x_vec: np.ndarray):
        x_mat = x_vec.reshape((self.m, self.n))
        if self.cov_chol is not None:
            Px = cho_solve(self.cov_chol, x_mat, check_finite=False).ravel()
        else:
            Px = (self.cov_inv @ x_mat).ravel()
        return Px

    def _matmat(self, x_mat: np.ndarray):
        K = x_mat.shape[1]
        x_reshaped = x_mat.reshape((self.m, self.n * K))
        if self.cov_chol is not None:
            Px = cho_solve(self.cov_chol, x_reshaped, check_finite=False)
        else:
            Px = self.cov_inv @ x_reshaped
        Px = Px.reshape((self.m, self.n, K)).reshape(self.m * self.n, K)
        return Px

    def _adjoint(self):
        return self

    def __reduce__(self):
        return (self.__class__, (self.cov_inv, self.cov_chol, self.n, self.m))
