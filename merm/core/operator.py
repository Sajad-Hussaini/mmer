import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve

class VLinearOperator(LinearOperator):
    """
    A linear operator that represents the marginal covariance matrix V.
    V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ + R
    """
    def __init__(self, random_effects, resid):
        self.random_effects = random_effects
        self.resid = resid
        super().__init__(dtype=np.float64, shape=(self.resid.m * self.resid.n, self.resid.m * self.resid.n))

    def _matvec(self, x_vec):
        """
        Computes the marginal covariance matrix-vector product V @ x_vec,
        where V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ + R.
        returns:
            1d array (Mn,)
        """
        Vx = self.resid.full_cov_matvec(x_vec)
        for re in self.random_effects.values():
            np.add(Vx, re.full_cov_matvec(x_vec), out=Vx)
        return Vx.ravel(order='F')

    def _adjoint(self):
        """Implements the adjoint operator Vᵀ. Since V is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.random_effects, self.resid))


class ResidualPreconditioner(LinearOperator):
    """
    The Lightweight Preconditioner: P⁻¹ = R⁻¹ = φ⁻¹ ⊗ Iₙ

    This is the simplest and cheapest preconditioner. It approximates the full
    covariance V with only its residual component R, ignoring all random effects.
    """
    def __init__(self, resid, n=None, m=None):
        if isinstance(resid, np.ndarray):
            self.n = n
            self.m = m
            self.cov_inv = resid
        else:
            self.n = resid.n
            self.m = resid.m
            self.cov_inv = solve(resid.cov, np.eye(self.m), assume_a='pos')

        super().__init__(dtype=np.float64, shape=(self.m * self.n, self.m * self.n))

    def _matvec(self, x_vec):
        """
        Computes (φ⁻¹ ⊗ Iₙ) @ x_vec.
        """
        x_mat = x_vec.reshape((self.n, self.m), order='F')
        Px = x_mat @ self.cov_inv
        return Px.ravel(order='F')
    
    def _adjoint(self):
        """ Since P is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.cov_inv, self.n, self.m))
