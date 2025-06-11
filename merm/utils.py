import numpy as np
from scipy import sparse

def cov_to_corr(cov):
    """
    Convert covariance matrix to correlation matrix.
    """
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)

def random_effect_design_matrix(group: np.ndarray, slope_covariates: np.ndarray = None):
    """
    Construct random effects design matrix for a grouping factor.
        Intercept block: one-hot encoding for group membership
        Slope block: covariate encoding for group membership
    Parameters:
        group: (n_samples,) array of group levels.
        slope_covariates: (n_samples, q) array for random slopes (optional).
    Returns:
        Z: Sparse matrix (n_samples, q * o).
        q: Number of random effects.
        o: Number of unique levels.
    """
    n = group.shape[0]
    levels, level_indices = np.unique(group, return_inverse=True)
    o = len(levels)
    intercept_block = sparse.csr_array((np.ones(n), (np.arange(n), level_indices)), shape=(n, o))
    blocks = [intercept_block]
    q = 1
    if slope_covariates is not None:
        q += slope_covariates.shape[1]
        for j in range(slope_covariates.shape[1]):
            data = slope_covariates[:, j]
            slope_block = sparse.csr_array((data, (np.arange(n), level_indices)), shape=(n, o))
            blocks.append(slope_block)
    Z = sparse.hstack(blocks, format='csr')
    return Z, q, o

def random_effect_design_matrices(X: np.ndarray, groups: np.ndarray, slope_cols: dict):
    """
    Construct random effects design matrices for multiple grouping factors.
    """
    Z, q, o = {}, {}, {}
    for k in range(groups.shape[1]):
        rsc_k = X[:, slope_cols[k]] if (slope_cols is not None and slope_cols[k] is not None) else None
        Z[k], q[k], o[k] = random_effect_design_matrix(groups[:, k], rsc_k)
    return Z, q, o

def block_diag_design_matrix(design_matrix: sparse.sparray, n_res: int):
    """
    Expands a design matrix into a block diagonal matrix using the Kronecker product.
    """
    return sparse.kron(sparse.eye_array(n_res, format='csr'), design_matrix, format='csr')

def block_diag_design_matrices(design_matrices: dict, n_res: int):
    """
    Create a dictionary of block diagonal design matrices for each group.
    """
    return {k: block_diag_design_matrix(Z, n_res) for k, Z in design_matrices.items()}

def crossprod_design_matrices(design_matrices: dict):
    """
    Compute the cross-product of design matrices for each group.
    """
    return {k: Z.T @ Z for k, Z in design_matrices.items()}
