import numpy as np
from joblib import Parallel, delayed, parallel_config
from .solver import BaseSolver

# ====================== Variance Correction ======================

class VarianceCorrection:
    """
    Variance correction computations.
    
    Provides unified interface for BSTE and DE correction methods,
    eliminating need for explicit method dispatch in M-step.
    """
    _VALID_METHODS = {'bste', 'de'}
    
    def __init__(self, method: str, cg_maxiter: int = 1000, n_jobs: int = -1, backend: str = 'threading'):
        """
        Initialize orchestrator.
        
        Parameters
        ----------
        method : str
            Correction method: 'bste' or 'de'.
        cg_maxiter : int, default=1000
            Maximum iterations for conjugate gradient solver.
        n_jobs : int, default=-1
            Number of parallel jobs to use.
        backend : str, default='threading'
            Joblib backend. 'threading' avoids memory duplication for sparse matrices.
        """
        if method not in self._VALID_METHODS:
            raise ValueError(f"Method must be one of {self._VALID_METHODS}, got {method}")
        self.method = method
        self.cg_maxiter = cg_maxiter
        self.n_jobs = n_jobs
        self.backend = backend
    
    def compute_correction(self, k: int, solver: BaseSolver, n_probes: int = None, iteration: int = 0) -> tuple:
        """
        Compute adaptive uncertainty correction terms (T, W).

        Parameters
        ----------
        k : int
            Index of random effect term.
        solver : BaseSolver
            Solver used for V@x updates.
        n_probes : int, optional
            Number of probes for stochastic methods. Auto-computed if None.
        iteration : int, default=0
            Current EM iteration index. Used to vary the random seed for probe
            generation so that each EM iteration draws independent probes,
            preventing systematic bias from identical probe vectors.

        Returns
        -------
        T : np.ndarray
            Trace correction matrix.
        W : np.ndarray
            Covariance correction matrix.
        """
        re = solver.realized_effects[k]
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
        if active_method == 'bste' and block_size <= n_probes:
            active_method = 'de'

        if active_method == 'de':
            return compute_cov_correction_de(k, solver, self.n_jobs, self.backend)
        elif active_method == 'bste':
            return compute_cov_correction_bste(k, solver, n_probes, self.n_jobs, self.backend, iteration)

def _generate_rademacher_probes(n_rows: int, n_probes: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Rademacher random matrix ({-1, +1}) as float64.

    Uses ``rng.integers(0, 2)`` to produce {0, 1} then maps in-place to
    {-1, +1}.  Avoids the zero edge-case from ``rng.random()`` (value 0.5
    maps to 0.0 after *=2;-=1, giving np.sign=0 not ±1).
    The *seed* should differ across EM iterations for independent draws.
    """
    rng = np.random.default_rng(seed)
    out = rng.integers(0, 2, size=(n_rows, n_probes), dtype=np.intp).astype(np.float64, copy=False)
    out *= 2.0   # {0,1} → {0,2}  in-place
    out -= 1.0   # {0,2} → {-1,+1} in-place
    return out

# ====================== Block Stochastic Trace Estimation ======================

def compute_cov_correction_bste(k: int, solver: BaseSolver, n_probes: int, n_jobs: int, backend: str, iteration: int = 0):
    """
    Compute adaptive uncertainty correction terms (T, W) using Block Stochastic
    Trace Estimation (BSTE).

    Parameters
    ----------
    iteration : int, default=0
        EM iteration index. Mixed into the per-column seed so that probe vectors
        differ across iterations, preventing systematic estimator bias.
    """
    m = solver.m
    re = solver.realized_effects[k]
    q = re.q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(
            delayed(_cov_correction_per_response_bste)(solver, n_probes, k, col, iteration)
            for col in range(m)
        )

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

def _cov_correction_per_response_bste(solver, n_probes: int, k: int, col: int, iteration: int = 0):
    """
    Compute BSTE correction terms (T, W) for a single response column.

    The seed is derived from *col* and *iteration* so that:
    - Columns are always independent of each other (col offset).
    - Iterations are independent of each other (iteration offset), preventing
      systematic bias from reusing identical probe vectors every EM step.
    """
    m = solver.m
    re = solver.realized_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col

    # Combine col and iteration into a single seed so probes are independent
    # both across columns and across EM iterations.
    seed = 42 + col + iteration * m
    vec = np.zeros((m * block_size, n_probes))
    probe_vectors = _generate_rademacher_probes(block_size, n_probes, seed)
    vec[col * block_size : (col + 1) * block_size, :] = probe_vectors
    
    vec_cg = re._kronZ_D_matvec(vec)
    vec_cg = solver.solve(vec_cg)
        
    lower_c = re._kronZ_D_T_matvec(vec_cg)[col * block_size:, :]
    # lower_c is (num_blocks * block_size, n_probes)
    # probe_vectors is (block_size, n_probes)
    # diag_C computation: 
    # For each block block_size, dot elementwise with probe_vectors, then sum over probes.
    # We want diag_C of length (num_blocks * block_size).
    # Reshape to (num_blocks, block_size, n_probes) and multiply by probe_vectors (1, block_size, n_probes)
    lower_c_reshaped = lower_c.reshape(num_blocks, block_size, n_probes)
    # einsum contracts (blocks, elements, probes) with (elements, probes) → (blocks, elements)
    # avoiding a (num_blocks, block_size, n_probes) broadcast intermediate
    diag_C = np.einsum('kbp,bp->kb', lower_c_reshaped, probe_vectors).ravel()

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

def compute_cov_correction_de(k: int, solver: BaseSolver, n_jobs: int, backend: str):
    """
    Compute adaptive uncertainty correction terms (T, W) for covariance updates using Deterministic Estimation (DE).
    """
    m = solver.m
    re = solver.realized_effects[k]
    q = re.q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(delayed(_cov_correction_per_response_de)(solver, k, col) for col in range(m))

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

def _cov_correction_per_response_de(solver, k: int, col: int):
    """
    Compute adaptive uncertainty correction terms (T, W) for covariance updates using Deterministic Estimation (DE) for a single response.
    """
    m = solver.m
    re = solver.realized_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col
    base_idx = col * block_size
    
    vec = np.zeros((m * block_size, block_size))
    # Identity matrix block
    vec[base_idx : base_idx + block_size, :] = np.eye(block_size)
    
    vec_cg = re._kronZ_D_matvec(vec)
    vec_cg = solver.solve(vec_cg)

    D_matvec_out = re._D_matvec(vec)
    kron_D_T_out = re._kronZ_D_T_matvec(vec_cg)
    lower_sigma = (D_matvec_out - kron_D_T_out)[col * block_size:]

    sigma_blocks = lower_sigma.reshape(num_blocks, block_size, block_size)
    # Cache ZTZ as dense once; einsum replaces the Python-level loop over
    # num_blocks individual .multiply().sum() calls.
    ZTZ_dense = re.ZTZ.toarray()
    T_traces = np.einsum('ij,kij->k', ZTZ_dense, sigma_blocks)
    W_blocks = lower_sigma.reshape(num_blocks, q, o, q, o).sum(axis=(2, 4))
    
    return col, T_traces, W_blocks
