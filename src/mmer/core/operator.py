from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator

from .corrections import VarianceCorrection, compute_cov_correction
from .terms import RealizedRandomEffect, RealizedResidual

__all__ = ["VarianceCorrection", "compute_cov_correction", "VLinearOperator", "ResidualPreconditioner"]


class VLinearOperator(LinearOperator):
    """
    Linear operator for the marginal covariance matrix V.

    Represents the matrix-vector product with:
    V = Σ (I_m ⊗ Z_k) D_k (I_m ⊗ Z_k)^T + R

    Wraps 'RealizedRandomEffect' and 'RealizedResidual' objects to compute
    V @ x efficiently without forming the dense matrix V.

    Parameters
    ----------
    random_effects : tuple of RealizedRandomEffect
        Realized random effect components.
    realized_residual : RealizedResidual
        Realized residual component.
    """

    def __init__(self, random_effects: tuple[RealizedRandomEffect, ...], realized_residual: RealizedResidual):
        self.random_effects = random_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m
        super().__init__(dtype=np.float64, shape=(self.m * self.n, self.m * self.n))

    def _matvec(self, x_vec: np.ndarray):
        """Compute V @ x_vec."""
        Vx = self.realized_residual._full_cov_matvec(x_vec)
        for re in self.random_effects:
            Vx += re._full_cov_matvec(x_vec)
        return Vx

    def _adjoint(self):
        return self

    def __reduce__(self):
        return (self.__class__, (self.random_effects, self.realized_residual))


class ResidualPreconditioner(LinearOperator):
    """
    Preconditioner based on the residual covariance (R).

    Computes M^{-1} @ x, where M approximation is R.
    Uses a cached Cholesky factor so the preconditioner applies R^{-1}
    without forming an explicit inverse.
    """

    def __init__(self, resid_cov: np.ndarray, n: int, m: int):
        self.cov = np.array(resid_cov, copy=True)
        self._chol = cho_factor(resid_cov, lower=True, check_finite=False)
        self.n = n
        self.m = m
        super().__init__(dtype=np.float64, shape=(self.m * self.n, self.m * self.n))

    def _matvec(self, x_vec: np.ndarray):
        return cho_solve(self._chol, x_vec.reshape((self.m, self.n)), check_finite=False).ravel()

    def _adjoint(self):
        return self

    def __reduce__(self):
        return (self.__class__, (self.cov, self.n, self.m))