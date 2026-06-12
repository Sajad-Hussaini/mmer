import numpy as np
from joblib import Parallel, delayed, parallel_config
from ..linalg.solver import BaseSolver
from ..linalg.operator import KroneckerMath, generate_rademacher_probes


class VarianceCorrection:
    """
    Variance correction for MMER algorithms.

    This class implements different methods for correcting the marginal covariance
    matrix of the random effects.

    Methods
    -------
    - bste : block stochastic trace estimator.
    - de : direct estimation.

    Attributes
    ---------
    method : str
        The correction method to use.
    cg_maxiter : int
        Maximum number of iterations for the conjugate gradient solver.
    n_jobs : int
        Number of jobs to use for parallel processing.
    backend : str
        The backend to use for parallel processing (e.g., "threading", "loky").
    """

    _VALID_METHODS = {"bste", "de"}

    def __init__(
        self,
        method: str,
        cg_maxiter: int = 1000,
        n_jobs: int = -1,
        backend: str = "threading",
    ):
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Method must be one of {self._VALID_METHODS}, got {method}"
            )
        self.method = method
        self.cg_maxiter = cg_maxiter
        self.n_jobs = n_jobs
        self.backend = backend

    def compute_correction(
        self, k: int, solver: BaseSolver, n_probes: int = None, iteration: int = 0
    ) -> tuple:
        cov = solver.random_covs[k]
        d = solver.random_designs[k]
        block_size = cov.n_effects * d.n_levels

        if n_probes is None:
            n_probes = 60

        active_method = self.method
        if active_method == "bste" and block_size <= n_probes:
            active_method = "de"

        if active_method == "de":
            return compute_cov_correction_de(k, solver, self.n_jobs, self.backend)
        elif active_method == "bste":
            return compute_cov_correction_bste(
                k, solver, n_probes, self.n_jobs, self.backend, iteration
            )


def _compute_cov_correction_generic(
    k: int, solver: BaseSolver, func, n_jobs: int, backend: str, *args
):
    m = solver.n_responses
    cov = solver.random_covs[k]
    q = cov.n_effects
    S = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(
            delayed(func)(solver, k, col, *args) for col in range(m)
        )

        for col, S_col, W_lower_blocks in results:
            S[:, col] = S_col
            for i, W_block in enumerate(W_lower_blocks):
                row = col + i
                r_slice = slice(row * q, (row + 1) * q)
                c_slice = slice(col * q, (col + 1) * q)
                W[r_slice, c_slice] = W_block
                if row != col:
                    W[c_slice, r_slice] = W_block.T

    # Exact residual variance correction (mathematically correct for multiple random effects)
    T_k = solver.resid_cov.matrix @ S
    T_k = (T_k + T_k.T) / 2.0
    W = (W + W.T) / 2.0
    return T_k, W


def compute_cov_correction_bste(
    k: int,
    solver: BaseSolver,
    n_probes: int,
    n_jobs: int,
    backend: str,
    iteration: int = 0,
):
    return _compute_cov_correction_generic(
        k,
        solver,
        _cov_correction_per_response_bste,
        n_jobs,
        backend,
        n_probes,
        iteration,
    )


def _cov_correction_per_response_bste(
    solver, k: int, col: int, n_probes: int, iteration: int = 0
):
    m = solver.n_responses
    n = solver.n_samples
    cov = solver.random_covs[k]
    d = solver.random_designs[k]
    q, o = cov.n_effects, d.n_levels
    block_size = q * o
    num_blocks = m - col

    seed = 42 + col + iteration * m
    probe_vectors = generate_rademacher_probes(block_size, n_probes, seed)

    probe_reshaped = probe_vectors.reshape(q, o * n_probes)
    D_col_block = cov.matrix[:, col * q : (col + 1) * q]
    Dx = D_col_block @ probe_reshaped
    # Single reshape instead of two sequential ones (both are O(1) views, but
    # this is cleaner and avoids creating the intermediate shape object).
    Dx = Dx.reshape(m * q * o, n_probes)

    vec_cg = KroneckerMath._kronZ_matvec(d.Z, Dx, m, q, o, n)
    vec_cg = solver.solve(vec_cg)

    A_k = KroneckerMath._kronZ_T_matvec(d.Z, vec_cg, m, q, o, n)
    A_k_blocks = A_k.reshape(m, block_size, n_probes)
    S_col = np.einsum("mbp,bp->m", A_k_blocks, probe_vectors) / n_probes

    kron_D_T_out = KroneckerMath._D_matvec(cov.matrix, A_k, m, q, o)
    lower_c = kron_D_T_out[col * block_size :, :]

    lower_c_reshaped = lower_c.reshape(num_blocks, q, o, n_probes)
    probe_vectors_reshaped = probe_vectors.reshape(q, o, n_probes)
    W_C = (
        np.einsum("kfip,gip->kfg", lower_c_reshaped, probe_vectors_reshaped) / n_probes
    )

    sub_cov = cov.matrix[col * q :, col * q : (col + 1) * q]
    D_blocks = sub_cov.reshape(num_blocks, q, q)
    W_blocks = D_blocks * o - W_C

    return col, S_col, W_blocks


def compute_cov_correction_de(k: int, solver: BaseSolver, n_jobs: int, backend: str):
    return _compute_cov_correction_generic(
        k, solver, _cov_correction_per_response_de, n_jobs, backend
    )


def _cov_correction_per_response_de(solver, k: int, col: int):
    m = solver.n_responses
    n = solver.n_samples
    cov = solver.random_covs[k]
    d = solver.random_designs[k]
    q, o = cov.n_effects, d.n_levels
    block_size = q * o
    num_blocks = m - col

    probe_vectors = np.eye(block_size)
    probe_reshaped = probe_vectors.reshape(q, o * block_size)
    D_col_block = cov.matrix[:, col * q : (col + 1) * q]
    Dx = D_col_block @ probe_reshaped
    D_matvec_out = Dx.reshape(m * q * o, block_size)

    vec_cg = KroneckerMath._kronZ_matvec(d.Z, D_matvec_out, m, q, o, n)
    vec_cg = solver.solve(vec_cg)

    A_k = KroneckerMath._kronZ_T_matvec(d.Z, vec_cg, m, q, o, n)
    A_k_blocks = A_k.reshape(m, block_size, block_size)
    S_col = np.trace(A_k_blocks, axis1=1, axis2=2)

    kron_D_T_out = KroneckerMath._D_matvec(cov.matrix, A_k, m, q, o)
    lower_sigma = (D_matvec_out - kron_D_T_out)[col * block_size :]

    W_blocks = lower_sigma.reshape(num_blocks, q, o, q, o).trace(axis1=2, axis2=4)

    return col, S_col, W_blocks
