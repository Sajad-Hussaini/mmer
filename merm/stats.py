import numpy as np
from scipy.sparse.linalg import cg
import warnings
from joblib import Parallel, delayed, cpu_count
from . import linalg_op

NJOBS = max(1, int(cpu_count() * 0.70))

def compute_mu(V_inv_eps, rand_effect):
    """
    Computes the random effect conditional mean μ.
    mu: 2d array (o, M*q)
    """
    rand_effect.mu = linalg_op.kronZ_D_T_matvec(V_inv_eps, rand_effect)

def resid_cov(rand_effect, V_op, M_op):
    """
    Computes the random effect contribution to the residual covariance matrix
    by constructing uncertainty correction T: M x M
    Uses symmetry of the covariance matrix to reduce computations.
    """
    M = rand_effect.n_res
    cov = np.zeros((M, M))
    use_parallel = M > 2

    if use_parallel:
        results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(resid_cov_correction)
                                                        (rand_effect, V_op, M_op, row, col)
                                                        for row in range(M) for col in range(row, M))
    else:
        results = [resid_cov_correction(rand_effect, V_op, M_op, row, col) for row in range(M) for col in range(row, M)]
    for row, col, trace in results:
        cov[col, row] = cov[row, col] = trace
    return cov

def rand_effect_cov(rand_effect, V_op, M_op):
    r"""
    Compute the random effect covariance matrix using
    $\hat{\tau}_k = \frac{1}{o_k}(U_k + W_k)$
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    # Compute indices for all levels
    m_idx = np.arange(M)[:, None]
    q_idx = np.arange(q)[None, :]
    base_idx = m_idx * q * o + q_idx * o

    beta = rand_effect.mu.reshape((o, M * q), order='F')
    U = beta.T @ beta

    use_parallel = o > 2
    if use_parallel:
        results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(re_cov_correction)
                                                        (rand_effect, V_op, M_op, (base_idx + j).ravel()) for j in range(o))
    else:
        results = [re_cov_correction(rand_effect, V_op, M_op, (base_idx + j).ravel()) for j in range(o)]

    W = np.sum(results, axis=0)
    rand_effect.cov = rand_effect.cov + (U - W) / o + 1e-6 * np.eye(M * q)

def resid_cov_correction(rand_effect, V_op, M_op, row, col):
    r"""
    Computes the trace of uncertainty correction matrix for the response block (row, col)
    $T_k_{ij} = \text{trace}\left( (Z_k^T Z_k) (\Sigma_k)_{ij} \right)$
    using the random effect conditional covariance
        Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    block_size = q * o

    tau_block = rand_effect.cov[row * q : (row + 1) * q, col * q : (col + 1) * q]
    D_block = np.kron(tau_block, np.eye(o))
    sigma_block = np.zeros((block_size, block_size))

    base_idx = col * block_size # starting basis col for extraction
    vec = np.zeros(M * block_size)
    for i in range(block_size):
        vec.fill(0.0)
        vec[base_idx + i] = 1.0
        rhs = linalg_op.kronZ_D_matvec(vec, rand_effect)
        x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
        rht = linalg_op.kronZ_D_T_matvec(x_sol, rand_effect)
        sigma_block[:, i] = rht.ravel(order='F')[row * block_size : (row + 1) * block_size]
    sigma_block =  D_block - sigma_block
    return row, col, np.sum(sigma_block * rand_effect.ZTZ)

def re_cov_correction(rand_effect, V_op, M_op, lvl_indices):
    r"""
    Computes the right hand term of random effect conditional covariance
        Σ = D - D (I_M ⊗ Z)^T V^{-1} (I_M ⊗ Z) D
    for the level block specified by lvl_indices.
    """
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    block_size = M * q

    sigma_block = np.zeros((block_size, block_size))

    vec = np.zeros(block_size * o)
    for i in range(block_size):
        vec.fill(0.0)
        vec[lvl_indices[i]] = 1.0
        rhs = linalg_op.kronZ_D_matvec(vec, rand_effect)
        x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
        rht = linalg_op.kronZ_D_T_matvec(x_sol, rand_effect)
        sigma_block[:, i] = rht.ravel(order='F')[lvl_indices]
    return sigma_block