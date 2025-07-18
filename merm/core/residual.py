import numpy as np

class Residual:
    """
    Residual class to handle residual computations in mixed effects models.
    It provides methods to compute covariance, and residuals, and perform matrix-vector operations.
    where ϵ = y - fx - Σ(Iₘ ⊗ Z)μ
    """
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.cov = np.eye(m)

    def compute_cov(self, eps, T):
        """
        Compute the residual covariance matrix.
            ɸ = (S + T) / n
        """
        S = eps.T @ eps
        for Tk in T.values():
            np.add(S, Tk, out=S)

        phi = S / self.n + 1e-6 * np.eye(self.m)
        return phi
    
    def full_cov_matvec(self, x_vec: np.ndarray) -> np.ndarray:
        """
        Computes the residual covariance matrix-vector product (φ ⊗ Iₙ) @ x_vec.
        returns:
            2d array (n, m)
        """
        x_mat = x_vec.reshape((self.n, self.m), order='F')
        Rx = x_mat @ self.cov
        return Rx
    
    def cov_to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        """
        std = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(std, std)