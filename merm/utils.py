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

def block_diag_design_matrix(n_res: int, design_matrix: sparse.sparray):
    """
    Expands a design matrix into a block diagonal matrix using the Kronecker product.
    """
    return sparse.kron(sparse.eye_array(n_res, format='csr'), design_matrix, format='csr')

def block_diag_design_matrices(n_res: int, design_matrices: dict):
    """
    Create a dictionary of block diagonal design matrices for each group.
    """
    return {k: block_diag_design_matrix(Z, n_res) for k, Z in design_matrices.items()}

def crossprod_design_matrices(design_matrices: dict):
    """
    Compute the cross-product of design matrices for each group.
    """
    return {k: Z.T @ Z for k, Z in design_matrices.items()}

def slq_logdet(V_op, dim, num_probes=30, m=50):
    """
    Approximate log(det(V)) using Stochastic Lanczos Quadrature (SLQ).
    V_op: LinearOperator for V
    dim: dimension of V
    num_probes: number of random vectors
    m: number of Lanczos steps
    """
    logdet_est = 0.0
    for _ in range(num_probes):
        v = np.random.choice([-1, 1], size=dim)
        v = v / np.linalg.norm(v)
        Q = np.zeros((dim, m+1))
        alpha = np.zeros(m)
        beta = np.zeros(m)
        Q[:, 0] = v
        for j in range(m):
            w = V_op @ Q[:, j]
            if j > 0:
                w -= beta[j-1] * Q[:, j-1]
            alpha[j] = np.dot(Q[:, j], w)
            w -= alpha[j] * Q[:, j]
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-10 or j == m-1:
                break
            Q[:, j+1] = w / beta[j]
        T = np.diag(alpha[:j+1]) + np.diag(beta[:j], 1) + np.diag(beta[:j], -1)
        eigvals, eigvecs = np.linalg.eigh(T)
        logdet_est += np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))
    return dim * logdet_est / num_probes
