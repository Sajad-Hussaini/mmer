import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from joblib import Parallel, delayed, cpu_count
import functools

NJOBS = max(1, int(cpu_count() * 0.70))

def cond_mean(V_inv_eps, rand_effect):
    """
    Computes the random effect conditional mean by leveraging the kronecker structure.
    """
    rand_effect.mu = W_T_matvec(V_inv_eps, rand_effect)
    return rand_effect

def resid_cov2(rand_effect, V_op):
    """
    Computes the random effect contribution to the residual covariance matrix
    by constructing $T_{m1,m2} = tr(\Sigma_{m1,m2} Z^T Z)$ for all m in M
    Uses symmetry of the covariance matrix to reduce computations.
    """
    cov = np.zeros((rand_effect.n_res, rand_effect.n_res))
    for row in range(rand_effect.n_res):
        for col in range(row, rand_effect.n_res):
            sigma_block = cond_cov_res_block(rand_effect, V_op, row, col)
            trace = np.sum(sigma_block * rand_effect.Z_crossprod)
            cov[col, row] = cov[row, col] = trace
    return cov

def resid_cov(rand_effect, V_op):
    """
    Computes the random effect contribution to the residual covariance matrix
    by constructing $T_{m1,m2} = tr(\Sigma_{m1,m2} Z^T Z)$ for all m in M
    Uses symmetry of the covariance matrix to reduce computations.
    """
    M = rand_effect.n_res
    cov = np.zeros((M, M))
    use_parallel = M > 2

    if use_parallel:
        results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(resid_cov_worker)
                                                        (rand_effect, V_op, row, col)
                                                        for row in range(M) for col in range(row, M))
    else:
        results = [resid_cov_worker(rand_effect, V_op, row, col) for row in range(M) for col in range(row, M)]
    for row, col, trace in results:
        cov[col, row] = cov[row, col] = trace
    return cov

def resid_cov_worker(rand_effect, V_op, row, col):
    """
    Worker function for parallel resid_cov computation.
    Computes one element of the covariance matrix.
    """
    sigma_block = cond_cov_res_block(rand_effect, V_op, row, col)
    return row, col, np.sum(sigma_block * rand_effect.Z_crossprod)

def rand_effect_cov2(rand_effect, V_op):
    """
    Compute the random effect covariance matrix using
    $\tau_k = \frac{1}{o_k} \sum_{j=1}^{o_k} \left( \mu_{k_j} \mu_{k_j}^T + \Sigma_{k_{jj}}  \right) + 10^{-6}I_{M\cdot q_k}$.
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    cov = np.zeros((M * q, M * q))

    # Compute indices for all levels
    m_idx = np.arange(M)[:, None]
    q_idx = np.arange(q)[None, :]
    base_idx = m_idx * q * o + q_idx * o
    
    for j in range(o):
        lvl_indices = (base_idx + j).ravel()
        mu_j = rand_effect.mu[lvl_indices]
        sigma_block = cond_cov_lvl_block(rand_effect, V_op, lvl_indices)
        cov += np.outer(mu_j, mu_j) + sigma_block
    cov = cov / o + 1e-6 * np.eye(M * q)
    return cov

def rand_effect_cov(rand_effect, V_op):
    """
    Compute the random effect covariance matrix using
    $\tau_k = \frac{1}{o_k} \sum_{j=1}^{o_k} \left( \mu_{k_j} \mu_{k_j}^T + \Sigma_{k_{jj}}  \right) + 10^{-6}I_{M\cdot q_k}$.
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    cov = np.zeros((M * q, M * q))

    # Compute indices for all levels
    m_idx = np.arange(M)[:, None]
    q_idx = np.arange(q)[None, :]
    base_idx = m_idx * q * o + q_idx * o

    use_parallel = o > 10
    if use_parallel:
        results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(rand_effect_cov_worker)
                                                        (rand_effect, V_op, (base_idx + j).ravel()) for j in range(o))
    else:
        results = [rand_effect_cov_worker(rand_effect, V_op, (base_idx + j).ravel()) for j in range(o)]

    cov = sum(results) / o + 1e-6 * np.eye(M * q)
    return cov

def rand_effect_cov_worker(rand_effect, V_op, lvl_indices):
    """
    Worker function for parallel resid_cov computation.
    Computes one element of the covariance matrix.
    """
    mu_j = rand_effect.mu[lvl_indices]
    sigma_block = cond_cov_lvl_block(rand_effect, V_op, lvl_indices)
    return np.outer(mu_j, mu_j) + sigma_block

def V_operator2(random_effects, resid_cov, M, n):
    """Returns a picklable LinearOperator for the V matrix."""
    return VLinearOperator(random_effects, resid_cov, M, n)

def create_picklable_V_operator(random_effects, resid_cov, M, n):
    """Creates a picklable LinearOperator using functools.partial"""
    matvec_func = functools.partial(V_matvec, 
                                   random_effects=random_effects, 
                                   resid_cov=resid_cov, 
                                   M=M, n=n)
    return LinearOperator(shape=(M*n, M*n), matvec=matvec_func)

def V_operator(random_effects, resid_cov, M, n):
    """Returns a picklable LinearOperator for the V matrix."""
    return create_picklable_V_operator(random_effects, resid_cov, M, n)

def V_matvec(x_vec, random_effects, resid_cov, M, n):
    """
    Computes the matrix-vector product V @ x_vec, where V = Σ(I_M ⊗ Z_k) D_k (I_M ⊗ Z_k)^T + R is the marginal covariance.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((M, n)).T
    Vx = x_mat @ resid_cov
    for re in random_effects.values():
        np.add(Vx, cov_matvec(x_vec, re), out=Vx)
    return Vx.T.ravel()  # (M*n, )

def cov_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product cov @ x_vec, where cov = (I_M ⊗ Z) D (I_M ⊗ Z)^T is
    random effect contribution to the marginal covariance.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    A_k = W_T_matvec(x_vec, rand_effect)
    A_k = A_k.reshape((rand_effect.n_res, rand_effect.n_effect * rand_effect.n_level)).T
    return rand_effect.Z_matrix @ A_k  # (n, M)

def cond_cov_res_block(rand_effect, V_op, row, col):
    """
    Computes the random effect conditional covariance
        Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
    for the response block specified by (row, col).
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    block_size = q * o

    tau_block = rand_effect.cov[row * q : (row + 1) * q, col * q : (col + 1) * q]
    D_block = np.kron(tau_block, np.eye(o))
    sigma_block = np.zeros((block_size, block_size))
    base_idx = col * block_size # Extracts columns in W_matvec

    for i in range(block_size):
        vec = np.zeros(M * block_size)
        vec[base_idx + i] = 1.0
        rhs = W_matvec(vec, rand_effect)
        x_sol, _ = cg(V_op, rhs)
        sigma_block[:, i] = W_T_matvec(x_sol, rand_effect)[row * block_size : (row + 1) * block_size]

    return D_block - sigma_block

def cond_cov_res_block2(rand_effect, V_op, row, col):
    """
    Computes the random effect conditional covariance
        Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
    for the response block specified by (row, col).
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    block_size = q * o

    tau_block = rand_effect.cov[row * q : (row + 1) * q, col * q : (col + 1) * q]
    D_block = np.kron(tau_block, np.eye(o))
    base_idx = col * block_size # for column indices in W_matvec

    use_parallel = block_size > 50
    args_list = [(M, block_size, base_idx+i, row, rand_effect, V_op) for i in range(block_size)]
    if use_parallel:
        results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(cond_cov_res_block_worker)(arg) for arg in args_list)
    else:
        results = [cond_cov_res_block_worker(arg) for arg in args_list]
    sigma_block = np.column_stack(results)
    return D_block - sigma_block

def cond_cov_res_block_worker(args):
    """
    Worker function for parallel cond_cov_res_block computation.
    This worker computes one column of the block.
    """
    M, block_size, lvl_idx, row, rand_effect, V_op = args
    vec = np.zeros(M * block_size)
    vec[lvl_idx] = 1.0
    rhs = W_matvec(vec, rand_effect)
    x_sol, _ = cg(V_op, rhs)
    return W_T_matvec(x_sol, rand_effect)[row * block_size : (row + 1) * block_size]

def cond_cov_lvl_block2(rand_effect, V_op, lvl_indices):
    """
    Computes the random effect conditional covariance
        Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
    for the level block specified by lvl_indices.
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    block_size = M * q

    D_block = rand_effect.cov

    use_parallel = block_size > 10
    args_list = [(i, o, block_size, lvl_indices, rand_effect, V_op) for i in range(block_size)]
    if use_parallel:
        results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(cond_cov_lvl_block_worker)(arg) for arg in args_list)
    else:
        results = [cond_cov_lvl_block_worker(arg) for arg in args_list]
    sigma_block = np.column_stack(results)
    return D_block - sigma_block

def cond_cov_lvl_block_worker(args):
    """
    Worker function for parallel cond_cov_lvl_block computation.
    This worker computes one column of the block.
    """
    i, o, block_size, lvl_indices, rand_effect, V_op = args
    vec = np.zeros(block_size * o)
    vec[lvl_indices[i]] = 1.0
    rhs = W_matvec(vec, rand_effect)
    x_sol, _ = cg(V_op, rhs)
    return W_T_matvec(x_sol, rand_effect)[lvl_indices]

def cond_cov_lvl_block(rand_effect, V_op, lvl_indices):
    """
    Computes the random effect conditional covariance
        Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
    for the level block specified by lvl_indices.
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    block_size = M * q

    D_block = rand_effect.cov
    sigma_block = np.zeros((block_size, block_size))

    for i in range(block_size):
        vec = np.zeros(block_size * o)
        vec[lvl_indices[i]] = 1.0
        rhs = W_matvec(vec, rand_effect)
        x_sol, _ = cg(V_op, rhs)
        sigma_block[:, i] = W_T_matvec(x_sol, rand_effect)[lvl_indices]

    return D_block - sigma_block

def W_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W @ x_vec, where W = (I_M ⊗ Z) D maps a vector from
    the random effects space (pre-weighted by D) to the observation space.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_res * rand_effect.n_effect, rand_effect.n_level)).T
    A_k = x_mat @ rand_effect.cov
    A_k = A_k.reshape((rand_effect.n_level, rand_effect.n_res, rand_effect.n_effect)).transpose(1, 2, 0).reshape((rand_effect.n_res, rand_effect.n_effect * rand_effect.n_level)).T
    B_k = rand_effect.Z_matrix @ A_k
    return B_k.T.ravel()  # (M*n, )

def W_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W^T @ x_vec, where W^T = D (I_M ⊗ Z)^T maps a vector from
    the observation space back to the random effects space (post-weighted by D).
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_res, rand_effect.n_obs)).T
    A_k = rand_effect.Z_matrix.T @ x_mat
    A_k = A_k.reshape((rand_effect.n_effect, rand_effect.n_level, rand_effect.n_res)).transpose(1, 2, 0).reshape((rand_effect.n_level, rand_effect.n_res * rand_effect.n_effect))
    B_k = A_k @ rand_effect.cov
    B_k = B_k.reshape((rand_effect.n_level, rand_effect.n_res, rand_effect.n_effect)).transpose(1, 2, 0).ravel()  # (M*q*o, )
    return B_k

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

def slq_logdet(V_op, dim, num_probes=30, m=50):
    """
    Approximate log(det(V)) using Stochastic Lanczos Quadrature (SLQ).
    V_op: LinearOperator for V
    dim: dimension of V
    num_probes: number of random vectors
    m: number of Lanczos steps
    """
    logdet_est = 0.0
    rng = np.random.default_rng(seed=42)
    for _ in range(num_probes):
        v = rng.choice([-1, 1], size=dim)
        v = v / np.linalg.norm(v)
        
        alpha = np.zeros(m)
        beta = np.zeros(m)

        q_prev = np.zeros(dim)
        q_cur = v.copy()

        for j in range(m):
            w = V_op @ q_cur
            if j > 0:
                w -= beta[j-1] * q_prev
            alpha[j] = np.dot(q_cur, w)
            w -= alpha[j] * q_cur
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-10:
                break
            q_prev, q_cur = q_cur, w / beta[j]

        T = np.diag(alpha[:j+1]) + np.diag(beta[:j], 1) + np.diag(beta[:j], -1)
        eigvals, eigvecs = np.linalg.eigh(T)
        logdet_est += np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))

    return dim * logdet_est / num_probes

def slq_probe(args):
    V_op, dim, m, seed = args
    rng = np.random.default_rng(seed)
    v = rng.choice([-1, 1], size=dim)
    v = v / np.linalg.norm(v)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    q_prev = np.zeros(dim)
    q_cur = v.copy()
    for j in range(m):
        w = V_op @ q_cur
        if j > 0:
            w -= beta[j-1] * q_prev
        alpha[j] = np.dot(q_cur, w)
        w -= alpha[j] * q_cur
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-10:
            break
        q_prev, q_cur = q_cur, w / beta[j]
    T = np.diag(alpha[:j+1]) + np.diag(beta[:j], 1) + np.diag(beta[:j], -1)
    eigvals, eigvecs = np.linalg.eigh(T)
    return np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))

def slq_logdet2(V_op, dim, num_probes=30, m=50):
    """
    Parallel SLQ logdet estimation.
    """
    seeds = np.random.SeedSequence(42).spawn(num_probes)
    args = [(V_op, dim, m, int(s.generate_state(1)[0])) for s in seeds]
    # chunksize = (num_probes + NJOBS - 1) // NJOBS

    results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(slq_probe)
                                                     (arg) for arg in args)
    logdet_est = sum(results)
    return dim * logdet_est / num_probes

class VLinearOperator(LinearOperator):
    def __init__(self, random_effects, resid_cov, M, n):
        self.random_effects = random_effects
        self.resid_cov = resid_cov
        self.M = M
        self.n = n
        shape = (M * n, M * n)
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x_vec):
        """Implements the matrix-vector product."""
        return V_matvec(x_vec, self.random_effects, self.resid_cov, self.M, self.n)
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.random_effects, self.resid_cov, self.M, self.n))


    # def __call__(self, x_vec):
    #     return V_matvec(x_vec, self.random_effects, self.resid_cov, self.M, self.n)