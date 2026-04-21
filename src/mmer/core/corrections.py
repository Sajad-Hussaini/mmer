import numpy as np
from scipy.sparse.linalg import cg
from joblib import Parallel, delayed, parallel_config
from typing import TYPE_CHECKING
from .solver import build_solver

if TYPE_CHECKING:
    from .operator import VLinearOperator, ResidualPreconditioner


# ====================== Variance Correction ======================

class VarianceCorrection:
    """
    Variance correction computations.
    
    Provides unified interface for STE, BSTE, and DE correction methods,
    eliminating need for explicit method dispatch in M-step.
    """
    _VALID_METHODS = {'ste', 'bste', 'de'}
    
    def __init__(self, method: str, cg_maxiter: int = 1000, n_jobs: int = -1, backend: str = 'loky'):
        """
        Initialize orchestrator.
        
        Parameters
        ----------
        method : str
            Correction method: 'ste', 'bste', or 'de'.
        cg_maxiter : int, default=1000
            Maximum iterations for conjugate gradient solver.
        n_jobs : int, default=-1
            Number of parallel jobs to use.
        backend : str, default='loky'
            Joblib backend.
        """
        if method not in self._VALID_METHODS:
            raise ValueError(f"Method must be one of {self._VALID_METHODS}, got {method}")
        self.method = method
        self.n_jobs = n_jobs
        self.backend = backend
        self.cg_maxiter = cg_maxiter
    
    def compute_correction(self, k: int, V_op: 'VLinearOperator', M_op: 'ResidualPreconditioner',
                          n_probes: int = None) -> tuple:
        """
        Compute adaptive uncertainty correction terms (T, W).
        
        Parameters
        ----------
        k : int
            Index of random effect term.
        V_op : VLinearOperator
            Marginal covariance linear operator.
        M_op : ResidualPreconditioner
            Optional preconditioner.
        n_probes : int, optional
            Number of probes for stochastic methods. Auto-computed if None.
        
        Returns
        -------
        T : np.ndarray
            Trace correction matrix.
        W : np.ndarray
            Covariance correction matrix.
        """
        re = V_op.random_effects[k]
        block_size = re.q * re.o
        
        # Hutchinson's trace estimator variance decays as O(1/sqrt(n_probes))
        # irrespective of block_size. A fixed target of 50-60 is usually optimal.
        if n_probes is None:
            n_probes = 60
            
        # Mathematical Crossover:
        # If the exact matrix dimension (block_size) is smaller than the number 
        # of stochastic probes, Deterministic Estimation (de) requires FEWER 
        # linear solves (exactly `block_size` solves) and has ZERO variance!
        active_method = self.method
        if active_method in ['ste', 'bste'] and block_size <= n_probes:
            active_method = 'de'
            
        if active_method == 'de':
            return compute_cov_correction_de(k, V_op, M_op, self.cg_maxiter, self.n_jobs, self.backend)
        elif active_method == 'bste':
            return compute_cov_correction_bste(k, V_op, M_op, n_probes, self.cg_maxiter, self.n_jobs, self.backend)
        elif active_method == 'ste':
            return compute_cov_correction_ste(k, V_op, M_op, n_probes, self.cg_maxiter, self.n_jobs, self.backend)

def _generate_rademacher_probes(n_rows, n_probes, seed=42):
    rng = np.random.default_rng(seed)
    probes = rng.integers(0, 2, size=(n_rows, n_probes), dtype=np.int8)
    probes *= 2
    probes -= 1
    return probes.astype(np.float64)

# ====================== Stochastic Trace Estimation for Correction ======================

def compute_cov_correction_ste(k: int, V_op: 'VLinearOperator', M_op: 'ResidualPreconditioner', n_probes: int, cg_maxiter: int, n_jobs: int, backend: str):
    """
    Compute adaptive uncertainty correction terms (T, W) for covariance updates using Stochastic Trace Estimation (STE).
    """
    m = V_op.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    
    probe_vectors = _generate_rademacher_probes(re.m * re.q * re.o, n_probes, 42)
    v1 = re._kronZ_D_matvec(probe_vectors)
    
    solver = build_solver(V_op.random_effects, V_op.realized_residual, M_op is not None, cg_maxiter)
    v2, _, _ = solver.solve(v1)
        
    v3 = re._kronZ_D_T_matvec(v2)
    diag_C = (probe_vectors * v3).sum(axis=1) / n_probes

    diag_sigma = np.repeat(np.diag(re.term.cov), o) 
    diag_sigma -= diag_C

    W_diag = diag_sigma.reshape((m, q, o)).sum(axis=2).ravel()
    sigma_mat  = diag_sigma.reshape(m, q * o)
    ZTZ_diag = re.ZTZ.diagonal()
    T_diag = sigma_mat.dot(ZTZ_diag)

    return np.diag(T_diag), np.diag(W_diag)

# ====================== Block Stochastic Trace Estimation ======================

def compute_cov_correction_bste(k: int, V_op: 'VLinearOperator', M_op: 'ResidualPreconditioner', n_probes: int, cg_maxiter: int, n_jobs: int, backend: str):
    """
    Compute adaptive uncertainty correction terms (T, W) for covariance updates using Block Stochastic Trace Estimation (BSTE).
    """
    m = V_op.m
    re = V_op.random_effects[k]
    q = re.q 
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    solver = build_solver(V_op.random_effects, V_op.realized_residual, M_op is not None, cg_maxiter)
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(delayed(_cov_correction_per_response_bste)(solver, n_probes, k, V_op, col) for col in range(m))

        for col, T_lower_traces, W_lower_diags in results:
            for i, (trace, W_diag) in enumerate(zip(T_lower_traces, W_lower_diags)):
                row = col + i
                T[col, row] = T[row, col] = trace
                r_slice = slice(row * q, (row + 1) * q)
                c_slice = slice(col * q, (col + 1) * q)
                np.fill_diagonal(W[r_slice, c_slice], W_diag)
                if row != col:
                    np.fill_diagonal(W[c_slice, r_slice], W_diag)
    return T, W

def _cov_correction_per_response_bste(solver, n_probes: int, k: int, V_op: 'VLinearOperator', col: int):
    """
    Compute adaptive uncertainty correction terms (T, W) for covariance updates using Block Stochastic Trace Estimation (BSTE) for a single response.
    """
    m = V_op.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col

    seed_base = 42
    vec = np.zeros((m * block_size, n_probes))
    probe_vectors = _generate_rademacher_probes(block_size, n_probes, seed_base)
    vec[col * block_size : (col + 1) * block_size, :] = probe_vectors
    
    vec_cg = re._kronZ_D_matvec(vec)
    vec_cg, _, _ = solver.solve(vec_cg)
        
    lower_c = re._kronZ_D_T_matvec(vec_cg)[col * block_size:, :]
    # lower_c is (num_blocks * block_size, n_probes)
    # probe_vectors is (block_size, n_probes)
    # diag_C computation: 
    # For each block block_size, dot elementwise with probe_vectors, then sum over probes.
    # We want diag_C of length (num_blocks * block_size).
    # Reshape to (num_blocks, block_size, n_probes) and multiply by probe_vectors (1, block_size, n_probes)
    lower_c_reshaped = lower_c.reshape(num_blocks, block_size, n_probes)
    diag_C = (lower_c_reshaped * probe_vectors[None, :, :]).sum(axis=2).ravel()

    diag_C /= n_probes
    sub_cov = re.term.cov[col*q:, col*q:(col+1)*q] # Use term.cov
    diags = sub_cov.reshape(num_blocks, q, q).diagonal(axis1=1, axis2=2)
    diag_sigma = np.repeat(diags, o, axis=1).ravel()
    diag_sigma -= diag_C

    ZTZ_diag = re.ZTZ.diagonal()
    sigma_mat  = diag_sigma.reshape(num_blocks, q * o)
    T_traces = sigma_mat.dot(ZTZ_diag)
    W_diag_blocks = diag_sigma.reshape(num_blocks, q, o).sum(axis=2)

    return col, T_traces, W_diag_blocks

# ====================== Deterministic Correction ======================

def compute_cov_correction_de(k: int, V_op: 'VLinearOperator', M_op: 'ResidualPreconditioner', cg_maxiter: int, n_jobs: int, backend: str):
    """
    Compute adaptive uncertainty correction terms (T, W) for covariance updates using Deterministic Estimation (DE).
    """
    m = V_op.m
    re = V_op.random_effects[k]
    q = re.q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    solver = build_solver(V_op.random_effects, V_op.realized_residual, M_op is not None, cg_maxiter)
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(delayed(_cov_correction_per_response_de)(solver, k, V_op, col) for col in range(m))

        for col, T_lower_traces, W_lower_blocks in results:
            for i, (trace, W_block) in enumerate(zip(T_lower_traces, W_lower_blocks)):
                row = col + i
                T[col, row] = T[row, col] = trace
                r_slice = slice(row * q, (row + 1) * q)
                c_slice = slice(col * q, (col + 1) * q)
                W[r_slice, c_slice] = W_block
                if row != col:
                     W[c_slice, r_slice] = W_block.T
    return T, W

def _cov_correction_per_response_de(solver, k: int, V_op: 'VLinearOperator', col: int):
    """
    Compute adaptive uncertainty correction terms (T, W) for covariance updates using Deterministic Estimation (DE) for a single response.
    """
    m = V_op.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col
    base_idx = col * block_size
    
    vec = np.zeros((m * block_size, block_size))
    # Identity matrix block
    vec[base_idx : base_idx + block_size, :] = np.eye(block_size)
    
    vec_cg = re._kronZ_D_matvec(vec)
    vec_cg, _, _ = solver.solve(vec_cg)

    D_matvec_out = re._D_matvec(vec)
    kron_D_T_out = re._kronZ_D_T_matvec(vec_cg)
    lower_sigma = (D_matvec_out - kron_D_T_out)[col * block_size:]

    sigma_blocks = lower_sigma.reshape(num_blocks, block_size, block_size)
    elementwise_prod = re.ZTZ.multiply(sigma_blocks)
    T_traces = elementwise_prod.sum(axis=(1, 2))
    W_blocks = lower_sigma.reshape(num_blocks, q, o, q, o).sum(axis=(2, 4))
    
    return col, T_traces, W_blocks
