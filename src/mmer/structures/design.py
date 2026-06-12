import weakref
import numpy as np
from scipy import sparse
from functools import cached_property


class RandomDesign:
    """
    Constructs and caches the sparse design matrix Z for a random effect.
    Purely a data structure, completely agnostic to covariance state or math operations.

    Parameters
    ----------
    group_data : np.ndarray
        1D array of grouping factors for this specific random effect, of shape (n,).
    covariates : np.ndarray or None, default=None
        2D array of covariates for random slopes, shape (n, p). None for intercept-only.

    Attributes
    ----------
    n : int
        Number of samples.
    Z : scipy.sparse.csr_array
        The sparse design matrix mapping observations to random effect levels.
    q : int
        The number of effects per level (1 for intercept, 1+p for slopes).
    o : int
        The number of unique levels in this grouping factor.
    ZTZ : scipy.sparse.csr_array
        The cached cross-product :math:`Z^T Z`.
    """

    def __init__(self, group_data: np.ndarray, covariates: np.ndarray | None = None):
        self.n_samples = group_data.shape[0]
        self.Z, self.n_effects, self.n_levels = self._build_Z(group_data, covariates)

        # Cache for exact DE corrections
        self.ZTZ = self.Z.T @ self.Z
        self.ZTZ_diag = self.ZTZ.diagonal().copy()
        self._cross_products_cache = weakref.WeakKeyDictionary()

    @staticmethod
    def _build_Z(group: np.ndarray, covariates: np.ndarray | None):
        n = group.shape[0]
        levels, level_indices = np.unique(group, return_inverse=True)
        o = len(levels)
        q = 1 if covariates is None else 1 + covariates.shape[1]

        total_nnz = n * q
        coo_data = np.empty(total_nnz)
        coo_rows = np.empty(total_nnz, dtype=np.intp)
        coo_cols = np.empty(total_nnz, dtype=np.intp)

        base_rows = np.arange(n, dtype=np.intp)
        coo_data[:n] = 1.0
        coo_rows[:n] = base_rows
        coo_cols[:n] = level_indices

        if covariates is not None:
            for col in range(covariates.shape[1]):
                sl = slice((col + 1) * n, (col + 2) * n)
                coo_data[sl] = covariates[:, col]
                coo_rows[sl] = base_rows
                coo_cols[sl] = level_indices + (col + 1) * o

        Z = sparse.csr_array((coo_data, (coo_rows, coo_cols)), shape=(n, q * o))
        return Z, q, o

    def get_Z_cross_product(self, other: "RandomDesign"):
        """
        Lazily compute and cache the cross product :math:`Z_i^T Z_j`.

        Parameters
        ----------
        other : RandomDesign
            The other random design structure to cross-multiply with.

        Returns
        -------
        scipy.sparse.spmatrix
            The sparse matrix representing :math:`Z_i^T Z_j`.
        """
        if self is other:
            return self.ZTZ
        if other not in self._cross_products_cache:
            self._cross_products_cache[other] = self.Z.T @ other.Z
        return self._cross_products_cache[other]

    @cached_property
    def ZTZ_dense(self) -> np.ndarray:
        return self.ZTZ.toarray()
