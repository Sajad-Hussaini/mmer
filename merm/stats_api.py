import numpy as np
from scipy.sparse.linalg import cg
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
