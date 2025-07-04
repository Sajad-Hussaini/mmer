import numpy as np
from scipy.sparse.linalg import cg
from joblib import Parallel, delayed
from . import linalg_op

def compute_mu(prec_resid, rand_effect):
    """
    Computes the random effect conditional mean μ as 2d array (o, M*q).
        prec_resid: precision-weighted residuals V⁻¹(y-fx)
    """
    rand_effect.mu[...] = linalg_op.kronZ_D_T_matvec(prec_resid, rand_effect)

def compute_resid_cov_correction(rand_effect, V_op, M_op, n_jobs):
    """
    Computes the random effect contribution to the residual covariance matrix φ
    by constructing the uncertainty correction matrix T: m x m
    Uses symmetry of the covariance matrix to reduce computations.
    """
    m = rand_effect.m
    cov = np.zeros((m, m))
    use_parallel = m > 2

    if use_parallel:
        results = Parallel(n_jobs, backend='loky')(delayed(resid_cov_correction)
                                                        (rand_effect, V_op, M_op, row, col)
                                                        for row in range(m) for col in range(row, m))
    else:
        results = [resid_cov_correction(rand_effect, V_op, M_op, row, col) for row in range(m) for col in range(row, m)]
    for row, col, trace in results:
        cov[col, row] = cov[row, col] = trace
    return cov

def compute_rand_effect_cov(rand_effect, V_op, M_op, n_jobs):
    """
    Compute the random effect covariance matrix τ = (U + W) / o
    """
    M, q, o = rand_effect.m, rand_effect.q, rand_effect.o
    # Compute indices for all levels
    m_idx = np.arange(M)[:, None]
    q_idx = np.arange(q)[None, :]
    base_idx = m_idx * q * o + q_idx * o

    beta = rand_effect.mu.reshape((o, M * q), order='F')
    U = beta.T @ beta

    use_parallel = o > 2
    if use_parallel:
        results = Parallel(n_jobs, backend='loky')(delayed(re_cov_correction)
                                                        (rand_effect, V_op, M_op, (base_idx + j).ravel()) for j in range(o))
    else:
        results = [re_cov_correction(rand_effect, V_op, M_op, (base_idx + j).ravel()) for j in range(o)]

    rh_term = np.sum(results, axis=0)
    rand_effect.cov = rand_effect.cov + (U - rh_term) / o + 1e-6 * np.eye(M * q)

def resid_cov_correction(rand_effect, V_op, M_op, row, col):
    """
    Computes the element of the uncertainty correction matrix T that is:
        Tᵢⱼ = trace((Zₖ⁻ᵀ Zₖ) Σᵢⱼ)
    using the random effect conditional covariance
        Σ = D - D (Iₘ ⊗ Z)⁻ᵀ V⁻¹ (Iₘ ⊗ Z) D
    """
    m, q, o = rand_effect.m, rand_effect.q, rand_effect.o
    block_size = q * o

    tau_block = rand_effect.cov[row * q : (row + 1) * q, col * q : (col + 1) * q]
    D_block = np.kron(tau_block, np.eye(o))
    sigma_block = np.zeros((block_size, block_size))

    base_idx = col * block_size # starting basis col for extraction
    vec = np.zeros(m * block_size)
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
    """
    Computes the right hand term of the random effect conditional covariance Σ:
        D (Iₘ ⊗ Z)⁻ᵀ V⁻¹ (Iₘ ⊗ Z) D
    for the level block specified by lvl_indices.
    """
    M, q, o = rand_effect.m, rand_effect.q, rand_effect.o
    block_size = M * q

    rh_term = np.zeros((block_size, block_size))

    vec = np.zeros(block_size * o)
    for i in range(block_size):
        vec.fill(0.0)
        vec[lvl_indices[i]] = 1.0
        rhs = linalg_op.kronZ_D_matvec(vec, rand_effect)
        x_sol, _ = cg(V_op, rhs.ravel(order='F'), M=M_op)
        rht = linalg_op.kronZ_D_T_matvec(x_sol, rand_effect)
        rh_term[:, i] = rht.ravel(order='F')[lvl_indices]
    return rh_term