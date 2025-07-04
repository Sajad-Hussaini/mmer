import numpy as np
from scipy import sparse

def design_rand_effect(group: np.ndarray, covariates: np.ndarray = None):
    """
    Construct random effects design matrix (Z) for a grouping variable.
        Intercept block: one-hot encoding for group membership
        Slope block: covariate encoding for group membership
    Parameters:
        group: (n_samples,) array of group levels.
        covariates: (n_samples, q) array for random slopes (optional).
    Returns:
        Z: Sparse ndarray (n_samples, q * o).
        q: Number of random effects.
        o: Number of unique levels.
    """
    n = group.shape[0]
    levels, level_indices = np.unique(group, return_inverse=True)
    o = len(levels)
    base_rows = np.arange(n)
    # Components for the first block (intercept)
    all_data = [np.ones(n)]
    all_rows = [base_rows]
    all_cols = [level_indices]
    q = 1
    # Components for the other block (slope)
    if covariates is not None:
        q += covariates.shape[1]
        for col in range(covariates.shape[1]):
            col_offset = (col + 1) * o
            
            all_data.append(covariates[:, col])
            all_rows.append(base_rows)
            all_cols.append(level_indices + col_offset)
    final_data = np.concatenate(all_data)
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    Z = sparse.csr_array((final_data, (final_rows, final_cols)), shape=(n, q * o))
    return Z, q, o

def cov_to_corr(cov):
    """
    Convert covariance matrix to correlation matrix.
    """
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)
