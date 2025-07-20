import numpy as np

class Residual:
    """
    Residual class to handle residual computations in mixed effects models.
    It provides methods to compute covariance, and residuals, and perform matrix-vector operations.
    where ϵ = y - fx - Σ(Iₘ ⊗ Z)μ
    """
    __slots__ = ('n', 'm', 'cov')
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.cov = np.eye(m)

    def compute_cov(self, eps: np.ndarray, T_sum: np.ndarray):
        """
        Compute the residual covariance matrix.
            ɸ = (S + T) / n
        """
        epsr = eps.reshape((self.m, self.n))
        S = epsr @ epsr.T
        np.add(S, T_sum, out=S)

        phi = S / self.n + 1e-6 * np.eye(self.m)
        return phi
    
    def full_cov_matvec(self, x_vec: np.ndarray):
        """
        Computes the residual covariance matrix-vector product (φ ⊗ Iₙ) @ x_vec.
        takes:
            1d array (mn,)
        returns:
            1d array (mn,)
        """
        return (self.cov @ x_vec.reshape((self.m, self.n))).ravel()
    
    def to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        """
        std = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(std, std)