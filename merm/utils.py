import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from scipy.linalg import eigh_tridiagonal
import warnings
from joblib import Parallel, delayed, cpu_count

NJOBS = max(1, int(cpu_count() * 0.70))

def cond_mean(V_inv_eps, rand_effect):
    """
    Computes the random effect conditional mean by leveraging the kronecker structure.
    """
    re_vec = kronZ_T_matvec(V_inv_eps, rand_effect)
    rand_effect.mu = cov_D_matvec(re_vec, rand_effect)

def resid_cov(rand_effect, V_op):
    """
    Computes the random effect contribution to the residual covariance matrix
    by constructing $T_{m1,m2} = tr(\Sigma_{m1,m2} Z^T Z)$ for all m_{i}
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
    Computes one element (row, col) of the covariance matrix.
    """
    sigma_block = cond_cov_res_block(rand_effect, V_op, row, col)
    return row, col, np.sum(sigma_block * rand_effect.Z_crossprod)

def rand_effect_cov(rand_effect, V_op):
    """
    Compute the random effect covariance matrix using
    $\tau_k = \frac{1}{o_k} \sum_{j=1}^{o_k} \left( \mu_{k_j} \mu_{k_j}^T + \Sigma_{k_{jj}}  \right) + 10^{-6}I_{M\cdot q_k}$.
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    # Compute indices for all levels
    m_idx = np.arange(M)[:, None]
    q_idx = np.arange(q)[None, :]
    base_idx = m_idx * q * o + q_idx * o

    use_parallel = o > 2
    if use_parallel:
        results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(rand_effect_cov_worker)
                                                        (rand_effect, V_op, (base_idx + j).ravel()) for j in range(o))
    else:
        results = [rand_effect_cov_worker(rand_effect, V_op, (base_idx + j).ravel()) for j in range(o)]

    cov = np.sum(results, axis=0) / o + 1e-6 * np.eye(M * q)
    return cov

def rand_effect_cov_worker(rand_effect, V_op, lvl_indices):
    """
    Worker function for parallel resid_cov computation.
    Computes one element of the covariance matrix.
    """
    mu = rand_effect.mu[lvl_indices]
    sigma_block = cond_cov_lvl_block(rand_effect, V_op, lvl_indices)
    return np.outer(mu, mu) + sigma_block

def V_matvec(x_vec, random_effects, resid_cov, M, n):
    """
    Computes the margianl covariance matrix-vector product V @ x_vec,
        where V = Σ(I_M ⊗ Z_k) D_k (I_M ⊗ Z_k)^T + R.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    Vx = cov_R_matvec(x_vec, resid_cov, M, n)
    for re in random_effects.values():
        np.add(Vx, cov_re_matvec(x_vec, re), out=Vx)
    return Vx  # (M*n, )

def cov_re_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product cov_re @ x_vec,
        where cov_re = (I_M ⊗ Z) D (I_M ⊗ Z)^T is random effect contribution to the marginal covariance.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    A_k = kronZ_T_matvec(x_vec, rand_effect)
    Dx = cov_D_matvec(A_k, rand_effect)
    B_k = kronZ_matvec(Dx, rand_effect)
    return B_k  # (M*n,)

def kronZ_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (I_M ⊗ Z) @ x_vec maps a vector from
    the random effects space to the observation space.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_res, rand_effect.n_effect * rand_effect.n_level)).T
    A_k = rand_effect.Z_matrix @ x_mat
    return A_k.T.ravel()  # (M*n, )

def kronZ_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (I_M ⊗ Z)^T @ x_vec maps a vector from
    the observation space back to the random effects space.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_res, rand_effect.n_obs)).T
    A_k = rand_effect.Z_matrix.T @ x_mat
    return A_k.T.ravel()  # (M*q*o, )

def cov_D_matvec(x_vec, rand_effect):
    """
    Computes the random effect covariance matrix-vector product (Tau ⊗ I_o) @ x_vec.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_res * rand_effect.n_effect, rand_effect.n_level)).T
    Dx =  x_mat @ rand_effect.cov
    return Dx.T.ravel()  # (M*q*o, )

def cov_R_matvec(x_vec, resid_cov, M, n):
    """
    Computes the residual covaraince matrix-vector product (Phi ⊗ I_n) @ x_vec.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((M, n)).T
    Rx = x_mat @ resid_cov
    return Rx.T.ravel()  # (M*n, )

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
    vec = np.zeros(M * block_size)
    for i in range(block_size):
        vec.fill(0.0)
        vec[base_idx + i] = 1.0
        rhs = kronZ_matvec(vec, rand_effect)
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
    sigma_block = np.zeros((block_size, block_size))
    base_idx = col * block_size # Extracts columns in W_matvec
    vec = np.zeros(M * block_size)
    for i in range(block_size):
        vec.fill(0.0)
        vec[base_idx + i] = 1.0
        rhs = W_matvec(vec, rand_effect)
        x_sol, _ = cg(V_op, rhs)
        sigma_block[:, i] = W_T_matvec(x_sol, rand_effect)[row * block_size : (row + 1) * block_size]

    return D_block - sigma_block

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
    vec = np.zeros(block_size * o)
    for i in range(block_size):
        vec.fill(0.0)
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

def slq_logdet2(V_op: LinearOperator, lanczos_steps: int = 50, num_probes: int = 30) -> float:
    """
    Estimates the log-determinant of a symmetric positive-definite operator V.

    This function uses the Stochastic Lanczos Quadrature (SLQ) method, which
    combines Hutchinson's trace estimator with Lanczos quadrature to provide a
    scalable, matrix-free estimate of log(det(V)).

    Parameters
    ----------
    V_op : scipy.sparse.linalg.LinearOperator
        The linear operator for the matrix V. It must represent a symmetric
        positive-definite matrix.
    lanczos_steps : int, optional
        The number of Lanczos iterations to perform for each probe vector.
        This controls the accuracy of the quadrature rule. Higher is more
        accurate but more computationally expensive. Default is 50.
    num_probes : int, optional
        The number of random probe vectors to use for the stochastic trace
        estimation. Higher is more accurate but more computationally expensive.
        Default is 30.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns
    -------
    float
        An estimate of the log-determinant of V.
    """
    dim = V_op.shape[0]
    logdet_est = 0.0
    rng = np.random.default_rng(seed=33)
    
    for _ in range(num_probes):
        # Use Rademacher distribution for the probe vector
        v = rng.choice([-1.0, 1.0], size=dim)

        # The core Lanczos iteration
        # This builds the tridiagonal matrix T implicitly
        alphas = np.zeros(lanczos_steps)
        betas = np.zeros(lanczos_steps - 1)
        
        q_prev = np.zeros(dim)
        q_cur = v / np.linalg.norm(v)
        
        for j in range(lanczos_steps):
            w = V_op @ q_cur
            alpha_j = np.dot(q_cur, w)
            alphas[j] = alpha_j

            if j < lanczos_steps - 1:
                w = w - alpha_j * q_cur - (betas[j-1] if j > 0 else 0.0) * q_prev
                beta_j = np.linalg.norm(w)
                if beta_j < 1e-10:
                    # Krylov subspace is exhausted, truncate and exit loop
                    lanczos_steps = j + 1
                    alphas = alphas[:lanczos_steps]
                    betas = betas[:lanczos_steps-1]
                    break
                betas[j] = beta_j
                q_prev, q_cur = q_cur, w / beta_j
    
        # Step 2: Use Gaussian quadrature with eigenvalues/vectors of T
        # This is faster and more memory efficient than forming the dense T matrix
        eigvals, eigvecs = eigh_tridiagonal(alphas, betas, eigvals_only=False)

        # Step 3: Apply the log function and sum with quadrature weights
        # The weights are the squared first elements of the eigenvectors of T
        # Clip eigenvalues at a small epsilon to prevent log(0) or log(<0)
        # due to floating point inaccuracies.
        eps = np.finfo(eigvals.dtype).eps
        if np.any(eigvals <= eps):
            warnings.warn("Lanczos-generated eigenvalues are non-positive. "
                          "Clipping at machine epsilon for log calculation. "
                          "This may indicate the operator is not SPD or "
                          "lanczos_steps is too small.")
            eigvals = np.maximum(eigvals, eps)
            
        logdet_est += np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))

    # The formula for Hutchinson's trace estimator with Rademacher vectors is
    # Tr(log(V)) ≈ sum(v_i^T log(V) v_i) / num_probes.
    # The quadrature term logdet_est approximates v^T log(V) v, but for a
    # normalized vector u = v / ||v||. Since ||v||^2 = dim, we must scale
    # the result. The full expression is dim * E[u^T log(V) u].
    return dim * logdet_est / num_probes

def slq_probe(V_op, lanczos_steps, seed):
    """
    Single probe for SLQ logdet estimation.
    """
    try:
        rng = np.random.default_rng(seed)
        dim = V_op.shape[0]
        v = rng.choice([-1.0, 1.0], size=dim)
        alphas = np.zeros(lanczos_steps)
        betas = np.zeros(lanczos_steps - 1)
        q_prev = np.zeros(dim)
        q_cur = v / np.linalg.norm(v)
        
        for j in range(lanczos_steps):
            w = V_op @ q_cur
            alpha_j = np.dot(q_cur, w)
            alphas[j] = alpha_j

            if j < lanczos_steps - 1:
                w = w - alpha_j * q_cur - (betas[j-1] if j > 0 else 0.0) * q_prev
                beta_j = np.linalg.norm(w)
                if beta_j < 1e-10:
                    # Krylov subspace is exhausted, truncate and exit loop
                    lanczos_steps = j + 1
                    alphas = alphas[:lanczos_steps]
                    betas = betas[:lanczos_steps-1]
                    break
                betas[j] = beta_j
                q_prev, q_cur = q_cur, w / beta_j

        eigvals, eigvecs = eigh_tridiagonal(alphas, betas, eigvals_only=False)
        eps = np.finfo(eigvals.dtype).eps
        if np.any(eigvals <= eps):
            warnings.warn("Lanczos-generated eigenvalues are non-positive. "
                            "Clipping at machine epsilon for log calculation. "
                            "This may indicate the operator is not SPD or "
                            "lanczos_steps is too small.")
            eigvals = np.maximum(eigvals, eps)

        return np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))
    except np.linalg.LinAlgError as e:
        # warnings.warn(f"Probe failed due to LinAlgError: {e}. Skipping.")
        return 0.0

def slq_logdet(V_op: LinearOperator, lanczos_steps: int = 50, num_probes: int = 30):
    """
    Parallel SLQ logdet estimation.
    """
    dim = V_op.shape[0]
    seeds = np.random.SeedSequence(42).spawn(num_probes)
    results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(slq_probe)
                                                     (V_op, lanczos_steps, int(s.generate_state(1)[0])) for s in seeds)
    logdet_est = np.sum(results)
    return dim * logdet_est / num_probes

def cov_to_corr(cov):
    """
    Convert covariance matrix to correlation matrix.
    """
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)

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
    
    def _adjoint(self):
        """Implements the adjoint operator V^T. Since V is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.random_effects, self.resid_cov, self.M, self.n))