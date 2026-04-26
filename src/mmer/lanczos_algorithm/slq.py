import numpy as np
from scipy.linalg import eigh_tridiagonal
from joblib import Parallel, delayed, parallel_config
from ..core.operator import VLinearOperator


def _rademacher(rng: np.random.Generator, size) -> np.ndarray:
    """
    Generate a Rademacher random vector/matrix ({-1, +1}) as float64.

    Uses ``rng.integers(0, 2, dtype=np.float64)`` to produce {0., 1.} directly,
    then maps to {-1., +1.} with two in-place ops.  Avoids the zero edge-case
    that ``rng.random()`` introduces when the value 0.5 is sampled (which gives
    0.0 after *=2; -=1, so np.sign returns 0 not ±1).
    """
    out = rng.integers(0, 2, size=size, dtype=np.intp).astype(np.float64, copy=False)
    out *= 2.0   # {0, 1} → {0, 2}  — in-place
    out -= 1.0   # {0, 2} → {-1, +1} — in-place, exact, no sign ambiguity
    return out


def slq_probes_block(V_op: VLinearOperator, lanczos_steps: int, seeds: np.ndarray):
    """
    Multiple probes for SLQ logdet estimation using matrix-matrix operations.

    Performs Lanczos iteration to estimate log(det(V)) using multiple probe
    vectors simultaneously to exploit V_op.matmat (block operations).

    Returns
    -------
    tuple[float, int]
        ``(sum_of_probe_estimates, n_valid_probes)``.  The caller accumulates
        across batches and applies the final ``dim``-scaling.
    """
    n_probes = len(seeds)
    dim = V_op.shape[0]

    # Generate probe vectors directly into pre-allocated columns (no per-column temp)
    V = np.empty((dim, n_probes), dtype=np.float64)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        col = V[:, i]                                          # view, no copy
        col[:] = rng.integers(0, 2, size=dim, dtype=np.intp)  # fill {0,1}
        col *= 2.0                                             # in-place
        col -= 1.0                                             # in-place → {-1,+1}

    # --- Block Lanczos Iteration ---
    alphas = np.zeros((lanczos_steps, n_probes))
    betas  = np.zeros((lanczos_steps - 1, n_probes))

    Q_prev = np.zeros((dim, n_probes))
    v_norms = np.linalg.norm(V, axis=0)
    Q_cur = V / v_norms            # V is no longer needed after this

    # Pre-allocate reusable buffer for reorthogonalisation products
    # (avoids one (dim × n_probes) temp per Lanczos step)
    _buf = np.empty((dim, n_probes))

    active_mask = np.ones(n_probes, dtype=bool)

    for j in range(lanczos_steps):
        W = V_op @ Q_cur   # unavoidable new allocation (LinearOperator has no `out=`)

        # alpha_j = diag(Q_cur^T W)  — einsum avoids (dim × n_probes) temp
        np.einsum('ij,ij->j', Q_cur, W, out=alphas[j])

        if j < lanczos_steps - 1:
            # W -= alpha_j * Q_cur  — true in-place via pre-allocated _buf
            np.multiply(Q_cur, alphas[j], out=_buf)
            W -= _buf

            if j > 0:
                # W -= beta_{j-1} * Q_prev  — reuse _buf
                np.multiply(Q_prev, betas[j - 1], out=_buf)
                W -= _buf

            beta_j = np.linalg.norm(W, axis=0)

            new_converged = (beta_j < 1e-12) & active_mask
            if np.any(new_converged):
                beta_j[new_converged] = 0.0
                active_mask[new_converged] = False

            betas[j] = beta_j

            safe_beta = np.where(active_mask, beta_j, 1.0)
            W /= safe_beta
            W[:, ~active_mask] = 0.0

            Q_prev = Q_cur
            Q_cur  = W

            if not np.any(active_mask):
                actual_steps = j + 1
                alphas = alphas[:actual_steps]
                betas  = betas[:actual_steps - 1]
                lanczos_steps = actual_steps
                break

    # Accumulate probe estimates
    total_est   = 0.0
    valid_probes = 0
    for i in range(n_probes):
        try:
            probe_steps = lanczos_steps
            for j in range(lanczos_steps - 1):
                if betas[j, i] < 1e-12:
                    probe_steps = j + 1
                    break

            alpha_i = alphas[:probe_steps, i]
            beta_i  = betas[:probe_steps - 1, i]

            eigvals, eigvecs = eigh_tridiagonal(alpha_i, beta_i, eigvals_only=False)
            valid = eigvals > 0
            if np.any(valid):
                total_est += np.sum(np.log(eigvals[valid]) * (eigvecs[0, valid] ** 2))
                valid_probes += 1
        except Exception:
            pass

    return total_est, valid_probes


def slq_probe(V_op: VLinearOperator, lanczos_steps: int, seed: int):
    """
    Single probe for SLQ logdet estimation.

    Performs Lanczos iteration to estimate log(det(V)) using a single probe
    vector from the Rademacher distribution ({-1, +1}^n).

    Parameters
    ----------
    V_op : VLinearOperator
        Symmetric positive-definite linear operator V.
    lanczos_steps : int
        Number of Lanczos iterations.
    seed : int
        Random seed for probe vector generation.

    Returns
    -------
    float
        Probe estimate: ``Σ log(λ_i) * e_i[0]²``.
    """
    dim = V_op.shape[0]
    rng = np.random.default_rng(seed)
    v   = rng.integers(0, 2, size=dim, dtype=np.intp).astype(np.float64, copy=False)
    v  *= 2.0; v -= 1.0     # {0,1} → {-1,+1} in-place, no zero edge-case

    alphas = np.zeros(lanczos_steps)
    betas  = np.zeros(lanczos_steps - 1)
    q_prev = np.zeros(dim)
    q_cur  = v / np.linalg.norm(v)

    # Pre-allocate reusable scalar-times-vector buffer
    _buf = np.empty(dim)

    for j in range(lanczos_steps):
        w       = V_op @ q_cur          # unavoidable new allocation
        alphas[j] = np.dot(q_cur, w)

        if j < lanczos_steps - 1:
            # w -= alpha * q_cur  — true in-place via _buf
            np.multiply(q_cur, alphas[j], out=_buf)
            w -= _buf

            if j > 0:
                # w -= beta * q_prev  — reuse _buf
                np.multiply(q_prev, betas[j - 1], out=_buf)
                w -= _buf

            beta_j = np.linalg.norm(w)
            if beta_j < 1e-12:
                lanczos_steps = j + 1
                alphas = alphas[:lanczos_steps]
                betas  = betas[:lanczos_steps - 1]
                break
            betas[j] = beta_j

            w /= beta_j
            q_prev = q_cur
            q_cur  = w

    try:
        eigvals, eigvecs = eigh_tridiagonal(alphas, betas, eigvals_only=False)
    except Exception:
        return 0.0

    valid = eigvals > 0
    if not np.any(valid):
        return 0.0

    return np.sum(np.log(eigvals[valid]) * (eigvecs[0, valid] ** 2))


def logdet(V_op: VLinearOperator, lanczos_steps: int, n_probes: int,
           n_jobs: int = -1, backend: str = 'threading', random_seed: int = 42):
    """
    Estimate log-determinant using Stochastic Lanczos Quadrature (SLQ).

    Parameters
    ----------
    V_op : VLinearOperator
        Symmetric positive-definite linear operator V.
    lanczos_steps : int
        Number of Lanczos iterations per probe.
    n_probes : int
        Number of probe vectors.
    n_jobs : int
        Number of parallel jobs.
    backend : str, default='threading'
        Joblib parallel backend. 'threading' is recommended to avoid memory
        duplication when the operator holds sparse matrices.
    random_seed : int, default=42
        Base random seed for reproducibility.

    Returns
    -------
    float
        Estimated log-determinant: log(det(V)).
    """
    dim = V_op.shape[0]

    # A Krylov subspace spans at most `dim` directions; cap both limits.
    lanczos_steps = min(lanczos_steps, dim)
    n_probes      = min(n_probes, dim)

    seeds = [int(s.generate_state(1)[0])
             for s in np.random.SeedSequence(random_seed).spawn(n_probes)]

    # Slice probes into blocks to limit peak memory (Q_cur, Q_prev, W each
    # occupy dim × block_size × 8 bytes).
    block_size = min(32, n_probes)

    if block_size < 2:
        # Single-probe parallel path
        with parallel_config(backend=backend, n_jobs=n_jobs):
            results = Parallel(return_as="generator")(
                delayed(slq_probe)(V_op, lanczos_steps, s) for s in seeds
            )
            logdet_est = sum(results)
        return dim * logdet_est / n_probes

    # Block path — each batch returns (sum, n_valid); accumulate weighted mean.
    batches = [seeds[i:min(i + block_size, n_probes)]
               for i in range(0, n_probes, block_size)]
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(
            delayed(slq_probes_block)(V_op, lanczos_steps, batch)
            for batch in batches
        )
        total_sum   = 0.0
        total_valid = 0
        for batch_sum, batch_valid in results:
            total_sum   += batch_sum
            total_valid += batch_valid

    if total_valid == 0:
        return 0.0

    return dim * total_sum / total_valid