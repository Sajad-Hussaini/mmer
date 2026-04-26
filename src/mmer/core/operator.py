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

    def _matvec(self, x_vec: np.ndarray):
        """Compute V @ x_vec."""
        # Residual part: (R \\otimes I_n) x
        Vx = self.realized_residual._full_cov_matvec(x_vec)

        # Random Effects parts
        for re in self.random_effects:
            re._full_cov_matvec(x_vec, out=Vx)
        return Vx

    def _matmat(self, x_mat: np.ndarray):
        """Compute V @ x_mat."""
        # Residual part
        Vx = self.realized_residual._full_cov_matvec(x_mat)

        # Random Effects parts
        for re in self.random_effects:
            re._full_cov_matvec(x_mat, out=Vx)
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
