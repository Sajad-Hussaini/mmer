import numpy as np

class Residual:
    """
    Residual class to handle residual computations in mixed effects models.
    
    It provides methods to compute covariance, and residuals, and perform matrix-vector operations.
    where ϵ = y - fx - Σ(Iₘ ⊗ Z)μ
    
    Parameters
    ----------
    n : int
        Number of observations.
    m : int
        Number of output dimensions.
    """
    __slots__ = ('n', 'm', 'cov')
    
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.cov = np.eye(m)

    def _compute_cov(self, eps: np.ndarray, T_sum: np.ndarray):
        """
        Compute the residual covariance matrix.
        
        ɸ = (S + T) / n
        
        Parameters
        ----------
        eps : np.ndarray
            Residual vector of shape (m*n,).
        T_sum : np.ndarray
            Sum matrix of shape (m, m).
        
        Returns
        -------
        np.ndarray
            Covariance matrix of shape (m, m).
        """
        epsr = eps.reshape((self.m, self.n))
        phi = epsr @ epsr.T
        phi += T_sum
        phi /= self.n
        phi[np.diag_indices_from(phi)] += 1e-5
        return phi
    
    def _full_cov_matvec(self, x_vec: np.ndarray):
        """
        Compute the residual covariance matrix-vector product.
        
        Computes (φ ⊗ Iₙ) @ x_vec.
        
        Parameters
        ----------
        x_vec : np.ndarray
            Input vector of shape (m*n,).
        
        Returns
        -------
        np.ndarray
            Result vector of shape (m*n,).
        """
        return (self.cov @ x_vec.reshape((self.m, self.n))).ravel()
    
    def to_corr(self):
        """
        Convert covariance matrix to correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix of shape (m, m).
        """
        std = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(std, std)