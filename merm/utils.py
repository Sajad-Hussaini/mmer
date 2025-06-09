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

def random_effect_design_matrices(X: np.ndarray, groups: np.ndarray, n_groups: int, slope_cols: dict):
    """
    Construct random effects design matrices for multiple grouping factors.
    """
    Z, q, o = {}, {}, {}
    for k in range(n_groups):
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

def marginal_covariance(phi, tau, n_obs, n_level, Im_Z):
    """
    Compute the marginal covariance matrix V and the random effects covariance matrices D.
    """
    D = {k: sparse.kron(tau[k], sparse.eye_array(n_level[k], format='csr'), format='csr') for k in tau}
    V = sparse.kron(phi, sparse.eye_array(n_obs, format='csr'), format='csr')
    for k in tau:
        V += Im_Z[k] @ D[k] @ Im_Z[k].T
    return V, D

def splu_decomposition(phi, tau, n_obs, n_level, Im_Z):
    """
    Compute the sparse LU decomposition of the marginal covariance matrix V
    """
    V, D = marginal_covariance(phi, tau, n_obs, n_level, Im_Z)
    return sparse.linalg.splu(V.tocsc()), D

def compute_mu(V_inv_eps, D, Im_Z):
    """
    Compute the conditional mean of the random effects.
    """
    return {k: D[k] @ Im_Z[k].T @ V_inv_eps for k in D}

def compute_V_inv_Im_Z_D(D, splu, Im_Z, k):
    """
    Compute the inverse of the marginal covariance matrix V multiplied by the design matrix Im_Z and the random effects covariance D.
    """
    Im_Z_D = Im_Z[k] @ D[k]
    return sparse.csr_array(splu.solve(Im_Z_D.toarray()))

def compute_sigma(D, splu, Im_Z):
    """
    Compute the conditional covariance of the random effects.
    """
    sigma = {}
    for k in D:
        V_inv_Im_Z_D = compute_V_inv_Im_Z_D(D, splu, Im_Z, k)
        sigma[k] = D[k] - D[k] @ Im_Z[k].T @ V_inv_Im_Z_D
    return sigma

def compute_logL(eps_marginal, V_inv_eps, splu, n_res, n_obs):
    """
    Compute the log-likelihood of the marginal distribution of the residuals (the marginal log-likelihood)
    """
    eps_marginal_stacked = eps_marginal.ravel(order='F')
    log_det_V = np.sum(np.log(np.abs(splu.U.diagonal())))
    ll = -(n_res * n_obs * np.log(2 * np.pi) + log_det_V + eps_marginal_stacked.T @ V_inv_eps) / 2
    return ll

def sum_random_effect(mu, Im_Z, n_res, n_obs):
    """
    Compute the sum of random effects contributions for all groups.
    """
    effect_sum = np.zeros((n_obs, n_res))
    for k in mu:
        effect_sum += (Im_Z[k] @ mu[k]).reshape((n_obs, n_res), order='F')
    return effect_sum