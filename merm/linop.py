import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve

class VLinearOperator(LinearOperator):
    """
    A linear operator that represents the marginal covariance matrix V.
    V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)⁻ᵀ + R
    """
    def __init__(self, random_effects, resid):
        self.random_effects = random_effects
        self.resid = resid
        super().__init__(dtype=np.float64, shape=(self.resid.m * self.resid.n, self.resid.m * self.resid.n))

    def _matvec(self, x_vec):
        """
        Computes the marginal covariance matrix-vector product V @ x_vec,
        where V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)⁻ᵀ + R.
        returns:
            1d array (Mn,)
        """
        Vx = self.resid.full_cov_matvec(x_vec)
        for re in self.random_effects.values():
            np.add(Vx, re.full_cov_matvec(x_vec), out=Vx)
        return Vx.ravel(order='F')

    def _adjoint(self):
        """Implements the adjoint operator V⁻ᵀ. Since V is symmetric, return self."""
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

class DiagonalPreconditioner(LinearOperator):
    """
    The Diagonal Preconditioner: P = diag(V)

    This is a general-purpose preconditioner P⁻¹ element-wise inversion.
    It captures the individual variance of every element while ignoring covariance.
    """
    def __init__(self, random_effects, resid=None):
        if isinstance(random_effects, np.ndarray):
            self.inv_V_diag = random_effects
            shape = (self.inv_V_diag.size, self.inv_V_diag.size)
        else:
            shape = (resid.m * resid.n, resid.m * resid.n)
            self.inv_V_diag = self._compute_V_diagonal(random_effects, resid)

        super().__init__(dtype=np.float64, shape=shape)
    
    @staticmethod
    def _compute_V_diagonal(random_effects, resid):
        """Calculates the inverse of the diagonal of the covariance matrix V."""
        diag_phi = np.diag(resid.cov)
        v_diag = np.tile(diag_phi, (resid.n, 1))
        for re in random_effects.values():
            Zk_sq = re.Z.power(2)
            diag_tau_k = np.diag(re.cov)
            diag_Dk = np.tile(diag_tau_k, re.o)
            diag_Dk_mat = diag_Dk.reshape((re.q * re.o, re.m), order='F')
            v_diag += Zk_sq @ diag_Dk_mat

        return 1.0 / (v_diag.ravel(order='F') + 1e-9)

    def _matvec(self, x_vec):
        """Applies P⁻¹ via element-wise multiplication."""
        return x_vec * self.inv_V_diag
    
    def _adjoint(self):
        """Since the operator is diagonal, it's symmetric."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.inv_V_diag,))