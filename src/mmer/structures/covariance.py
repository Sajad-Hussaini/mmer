import numpy as np
from scipy.linalg import cholesky, LinAlgError


def make_pd(mat: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest positive-definite matrix.

    This function computes the eigendecomposition of the input matrix, thresholds
    the eigenvalues to ensure they are strictly positive, and reconstructs the matrix.

    Parameters
    ----------
    mat : np.ndarray
        A symmetric square matrix of shape (n, n).
    min_eig : float, default=1e-8
        The minimum allowable eigenvalue. All eigenvalues smaller than this
        threshold are clamped.

    Returns
    -------
    np.ndarray
        The projected positive-definite matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    floor = max(min_eig, min_eig * float(eigvals[-1]))
    np.maximum(eigvals, floor, out=eigvals)
    eigvecs *= np.sqrt(eigvals)  # (V√Λ)(V√Λ)ᵀ = VΛVᵀ — correct PD projection
    return eigvecs @ eigvecs.T


def ensure_pd(mat: np.ndarray) -> np.ndarray:
    """
    Cheaply check positive-definiteness via Cholesky; project only on failure.

    Attempts a fast Cholesky decomposition. If it fails due to negative or zero
    eigenvalues (LinAlgError), it falls back to the more expensive `make_pd`
    projection.

    Parameters
    ----------
    mat : np.ndarray
        A symmetric square matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        A guaranteed positive-definite matrix.
    """
    try:
        cholesky(mat, lower=True, check_finite=False)
        return mat.copy()
    except LinAlgError:
        return make_pd(mat)


class Covariance:
    """
    Learned state of a covariance matrix.

    This base class holds a dense covariance matrix and manages its updates,
    ensuring that it remains positive-definite after each update step.

    Parameters
    ----------
    size : int
        The dimension of the square covariance matrix (size x size).

    Attributes
    ----------
    matrix : np.ndarray
        The current positive-definite covariance matrix.
    """

    def __init__(self, size: int):
        self.size = size
        self.matrix = np.eye(size)

    def update(self, new_cov: np.ndarray):
        if new_cov.shape != (self.size, self.size):
            raise ValueError(
                f"Shape mismatch. Expected {(self.size, self.size)}, got {new_cov.shape}"
            )
        self.matrix = ensure_pd(new_cov)

    @property
    def correlation(self) -> np.ndarray:
        std = np.sqrt(np.diag(self.matrix))
        std[std == 0] = 1e-12
        return self.matrix / np.outer(std, std)


class RandomCovariance(Covariance):
    """
    State container for a Random Effects covariance matrix (G).

    Parameters
    ----------
    n_responses : int
        The number of response variables.
    n_effects : int
        The number of random effects per group level (e.g., 1 for intercept, 2 for intercept+slope).
    """

    def __init__(self, n_responses: int, n_effects: int):
        super().__init__(n_responses * n_effects)
        self.n_responses = n_responses
        self.n_effects = n_effects


class ResidualCovariance(Covariance):
    """
    State container for a Residual covariance matrix (R).

    Parameters
    ----------
    n_responses : int
        The number of response variables.
    """

    def __init__(self, n_responses: int):
        super().__init__(n_responses)
        self.n_responses = n_responses
