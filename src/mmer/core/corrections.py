from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, parallel_config

from .linear_solver import cg_solve


class VarianceCorrection:
    """
    Covariance-correction engine for STE, BSTE, and DE updates.

    The engine owns the method choice and the solver defaults so the estimator
    does not have to thread CG settings through every correction call.
    """

    _VALID_METHODS = {"ste", "bste", "de"}

    def __init__(
        self,
        method: str,
        n_probes: int | None = None,
        n_jobs: int = -1,
        backend: str = "loky",
        cg_rtol: float = 1e-5,
        cg_atol: float = 0.0,
        cg_maxiter: int | None = None,
    ):
        if method not in self._VALID_METHODS:
            raise ValueError(f"Method must be one of {self._VALID_METHODS}, got {method}")

        self.method = method
        self.n_probes = n_probes
        self.n_jobs = n_jobs
        self.backend = backend
        self.cg_rtol = cg_rtol
        self.cg_atol = cg_atol
        self.cg_maxiter = cg_maxiter

    def compute_correction(
        self,
        k: int,
        V_op: "VLinearOperator",
        M_op: "ResidualPreconditioner",
        n_probes: int | None = None,
        n_jobs: int | None = None,
        backend: str | None = None,
        cg_rtol: float | None = None,
        cg_atol: float | None = None,
        cg_maxiter: int | None = None,
    ) -> tuple:
        if n_probes is None:
            n_probes = self.n_probes
        if n_jobs is None:
            n_jobs = self.n_jobs
        if backend is None:
            backend = self.backend
        if cg_rtol is None:
            cg_rtol = self.cg_rtol
        if cg_atol is None:
            cg_atol = self.cg_atol
        if cg_maxiter is None:
            cg_maxiter = self.cg_maxiter

        if n_probes is None:
            re = V_op.random_effects[k]
            block_size = re.q * re.o
            n_probes = min(max(20, block_size // 10), 100)

        if self.method == "de":
            return compute_cov_correction_de(k, V_op, M_op, n_jobs, backend, cg_rtol, cg_atol, cg_maxiter)
        if self.method == "bste":
            return compute_cov_correction_bste(k, V_op, M_op, n_probes, n_jobs, backend, cg_rtol, cg_atol, cg_maxiter)
        return compute_cov_correction_ste(k, V_op, M_op, n_probes, n_jobs, backend, cg_rtol, cg_atol, cg_maxiter)


def compute_cov_correction(
    k: int,
    V_op: "VLinearOperator",
    M_op: "ResidualPreconditioner",
    method: str,
    n_jobs: int,
    backend: str,
    cg_rtol: float = 1e-5,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    n_probes: int | None = None,
) -> tuple:
    orchestrator = VarianceCorrection(
        method,
        n_probes=n_probes,
        n_jobs=n_jobs,
        backend=backend,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    return orchestrator.compute_correction(k, V_op, M_op)


def _generate_rademacher_probes(n_rows, n_probes, seed=42):
    rng = np.random.default_rng(seed)
    probes = rng.integers(0, 2, size=(n_rows, n_probes), dtype=np.int8)
    probes *= 2
    probes -= 1
    return probes.astype(np.float64)


def compute_cov_correction_ste(
    k: int,
    V_op: "VLinearOperator",
    M_op: "ResidualPreconditioner",
    n_probes: int,
    n_jobs: int,
    backend: str,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
):
    m = V_op.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    diag_C = np.zeros(m * q * o)
    with parallel_config(backend=backend, n_jobs=n_jobs):
        pcp_gen = Parallel(return_as="generator")(
            delayed(_compute_C_probe)(i, k, V_op, M_op, cg_rtol, cg_atol, cg_maxiter)
            for i in range(n_probes)
        )
        for diag_pcp in pcp_gen:
            diag_C += diag_pcp

    diag_C /= n_probes
    diag_sigma = np.repeat(np.diag(re.term.cov), o)
    diag_sigma -= diag_C

    W_diag = diag_sigma.reshape((m, q, o)).sum(axis=2).ravel()
    sigma_mat = diag_sigma.reshape(m, q * o)
    ZTZ_diag = re.layout.ztz_diag
    T_diag = sigma_mat.dot(ZTZ_diag)

    return np.diag(T_diag), np.diag(W_diag)


def _compute_C_probe(
    probe_index: int,
    k: int,
    V_op: "VLinearOperator",
    M_op: "ResidualPreconditioner",
    cg_rtol: float = 1e-5,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    x0: np.ndarray | None = None,
):
    seed = 42 + probe_index
    re = V_op.random_effects[k]
    probe_vector = _generate_rademacher_probes(re.m * re.q * re.o, 1, seed).ravel()
    v1 = re._kronZ_D_matvec(probe_vector)
    v2 = cg_solve(V_op, v1, M_op, x0=x0, rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter)
    probe_vector *= re._kronZ_D_T_matvec(v2)
    return probe_vector


def compute_cov_correction_bste(
    k: int,
    V_op: "VLinearOperator",
    M_op: "ResidualPreconditioner",
    n_probes: int,
    n_jobs: int,
    backend: str,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
):
    m = V_op.m
    re = V_op.random_effects[k]
    q = re.q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(
            delayed(_cov_correction_per_response_bste)(n_probes, k, V_op, M_op, col, cg_rtol, cg_atol, cg_maxiter)
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


def _cov_correction_per_response_bste(
    n_probes: int,
    k: int,
    V_op: "VLinearOperator",
    M_op: "ResidualPreconditioner",
    col: int,
    cg_rtol: float = 1e-5,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
):
    m = V_op.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col

    diag_C = np.zeros(num_blocks * block_size)
    vec = np.zeros(m * block_size)
    previous_solution = None
    for i in range(n_probes):
        seed = 42 + i
        probe_vector = _generate_rademacher_probes(block_size, 1, seed).ravel()
        vec[col * block_size : (col + 1) * block_size] = probe_vector
        vec_cg = re._kronZ_D_matvec(vec)
        vec_cg = cg_solve(V_op, vec_cg, M_op, x0=previous_solution, rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter)
        previous_solution = vec_cg
        lower_c = re._kronZ_D_T_matvec(vec_cg)[col * block_size :]
        vec[col * block_size : (col + 1) * block_size] = 0
        diag_C += (lower_c.reshape(num_blocks, block_size) * probe_vector).ravel()

    diag_C /= n_probes
    sub_cov = re.term.cov[col * q : , col * q : (col + 1) * q]
    diags = sub_cov.reshape(num_blocks, q, q).diagonal(axis1=1, axis2=2)
    diag_sigma = np.repeat(diags, o, axis=1).ravel()
    diag_sigma -= diag_C

    ZTZ_diag = re.layout.ztz_diag
    sigma_mat = diag_sigma.reshape(num_blocks, q * o)
    T_traces = sigma_mat.dot(ZTZ_diag)
    W_diag_blocks = diag_sigma.reshape(num_blocks, q, o).sum(axis=2)

    return col, T_traces, W_diag_blocks


def compute_cov_correction_de(
    k: int,
    V_op: "VLinearOperator",
    M_op: "ResidualPreconditioner",
    n_jobs: int,
    backend: str,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
):
    m = V_op.m
    re = V_op.random_effects[k]
    q = re.q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(
            delayed(_cov_correction_per_response_de)(k, V_op, M_op, col, cg_rtol, cg_atol, cg_maxiter)
            for col in range(m)
        )

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


def _cov_correction_per_response_de(
    k: int,
    V_op: "VLinearOperator",
    M_op: "ResidualPreconditioner",
    col: int,
    cg_rtol: float = 1e-5,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
):
    m = V_op.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col
    lower_sigma = np.empty((num_blocks * block_size, block_size))
    base_idx = col * block_size
    vec = np.zeros(m * block_size)
    previous_solution = None
    for i in range(block_size):
        vec[base_idx + i] = 1.0
        vec_cg = re._kronZ_D_matvec(vec)
        vec_cg = cg_solve(V_op, vec_cg, M_op, x0=previous_solution, rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter)
        previous_solution = vec_cg
        lower_sigma[:, i] = (re._D_matvec(vec) - re._kronZ_D_T_matvec(vec_cg))[col * block_size :]
        vec[base_idx + i] = 0

    sigma_blocks = lower_sigma.reshape(num_blocks, block_size, block_size)
    T_traces = re.layout.trace_against_block_stack(sigma_blocks)
    W_blocks = lower_sigma.reshape(num_blocks, q, o, q, o).sum(axis=(2, 4))

    return col, T_traces, W_blocks