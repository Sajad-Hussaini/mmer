import numpy as np

class Residual:
    """
    Residual class to handle residual computations in mixed effects models.
    It provides methods to compute covariance, and residuals, and perform matrix-vector operations.
    """
    
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.cov = np.eye(m)
        self.eps = np.zeros((n, m))
    
    def compute_eps(self, resid_mrg: np.ndarray, total_rand_effect: np.ndarray) -> np.ndarray:
        """
        Compute the conditional residuals:
            ϵ = y - fx - Σ(Iₘ ⊗ Z)μ
        returns:
            2d array (n, m)
        """
        np.subtract(resid_mrg, total_rand_effect, out=self.eps)
        return self

    def compute_cov(self, T):
        """
        Compute the residual covariance matrix.
            ɸ = (S + T) / n
        """
        S = self.eps.T @ self.eps
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