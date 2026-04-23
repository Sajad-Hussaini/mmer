import numpy as np
from scipy.linalg import eigh_tridiagonal
from joblib import Parallel, delayed, parallel_config
from ..core.operator import VLinearOperator


def slq_probes_block(V_op: VLinearOperator, lanczos_steps: int, seeds: np.ndarray):
    """
    Multiple probes for SLQ logdet estimation using matrix-matrix operations.
    
    Performs Lanczos iteration to estimate log(det(V)) using multiple probe vectors
    simultaneously to take advantage of V_op.matmat (block operations).
    """
    n_probes = len(seeds)
    dim = V_op.shape[0]
    
    # Generate all probe vectors
    V = np.empty((dim, n_probes), dtype=np.float64)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        v = rng.integers(0, 2, size=dim, dtype=np.int8)
        v *= 2
        v -= 1
        V[:, i] = v

    # --- Block Lanczos Iteration ---
    alphas = np.zeros((lanczos_steps, n_probes))
    betas = np.zeros((lanczos_steps - 1, n_probes))
    
    Q_prev = np.zeros((dim, n_probes))
    v_norms = np.linalg.norm(V, axis=0)
    Q_cur = V / v_norms
    
    # Keep track of active probes to avoid division by zero on early termination
    active_mask = np.ones(n_probes, dtype=bool)
    
    for j in range(lanczos_steps):
        W = V_op @ Q_cur
        
        # alpha_j = np.sum(Q_cur * W, axis=0)
        alphas[j, :] = np.sum(Q_cur * W, axis=0)
        
        if j < lanczos_steps - 1:
            W -= alphas[j, :] * Q_cur
            if j > 0:
                W -= betas[j-1, :] * Q_prev
                
            beta_j = np.linalg.norm(W, axis=0)
            
            # Mask out probes that have converged
            new_converged = (beta_j < 1e-12) & active_mask
            if np.any(new_converged):
                beta_j[new_converged] = 0.0
                active_mask[new_converged] = False
                
            betas[j, :] = beta_j
            
            # Avoid division by zero
            safe_beta = np.where(active_mask, beta_j, 1.0)
            W /= safe_beta
            W[:, ~active_mask] = 0.0
            
            Q_prev = Q_cur
            Q_cur = W
            
            if not np.any(active_mask):
                # All probes converged
                actual_steps = j + 1
                alphas = alphas[:actual_steps, :]
                betas = betas[:actual_steps - 1, :]
                lanczos_steps = actual_steps
                break

    # Process each probe's tridiagonal matrix
    total_est = 0.0
    for i in range(n_probes):
        try:
            # For each probe, find the actual number of steps it took before converging
            probe_steps = lanczos_steps
            for j in range(lanczos_steps - 1):
                if betas[j, i] < 1e-12:
                    probe_steps = j + 1
                    break
                    
            alpha_i = alphas[:probe_steps, i]
            beta_i = betas[:probe_steps-1, i]
            
            eigvals, eigvecs = eigh_tridiagonal(alpha_i, beta_i, eigvals_only=False)
            valid = eigvals > 0
            if np.any(valid):
                total_est += np.sum(np.log(eigvals[valid]) * (eigvecs[0, valid] ** 2))
        except Exception:
            pass

    return total_est


def slq_probe(V_op: VLinearOperator, lanczos_steps: int, seed: int):
    """
    Single probe for SLQ logdet estimation.
    
    Performs Lanczos iteration to estimate log(det(V)) using a single probe vector.
    Uses Rademacher distribution for probe vector v ~ {-1, 1}^n.
    
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
        Probe estimate: sum(log(λ_i) * (e_i[0])^2) where λ_i are eigenvalues
        of tridiagonal matrix T and e_i are corresponding eigenvectors.
    """
    rng = np.random.default_rng(seed)
    dim = V_op.shape[0]
    # Use Rademacher distribution for the probe vector ({-1, 1})
    v = rng.integers(0, 2, size=dim, dtype=np.int8)
    v *= 2
    v -= 1
    v = v.astype(np.float64)
    
    # --- Lanczos Iteration ---
    alphas = np.zeros(lanczos_steps)
    betas = np.zeros(lanczos_steps - 1)
    q_prev = np.zeros(dim)
    q_cur = v / np.linalg.norm(v)
    
    for j in range(lanczos_steps):
        w = V_op @ q_cur
        alpha_j = np.dot(q_cur, w)
        alphas[j] = alpha_j

        if j < lanczos_steps - 1:
            w -= alpha_j * q_cur
            if j > 0:
                w -= betas[j-1] * q_prev
            
            beta_j = np.linalg.norm(w)
            # Use a strict threshold to avoid numerical issues and stop early if needed
            if beta_j < 1e-12:
                lanczos_steps = j + 1
                alphas = alphas[:lanczos_steps]
                betas = betas[:lanczos_steps-1]
                break
            betas[j] = beta_j
            
            w /= beta_j
            q_prev = q_cur
            q_cur = w

    # Only wrap the potentially unstable eigenvalue computation
    try:
        # Compute eigenvalues and eigenvectors of the tridiagonal matrix T
        eigvals, eigvecs = eigh_tridiagonal(alphas, betas, eigvals_only=False)
    except Exception:
        # If eigenvalue computation fails (e.g., due to an unstable T),
        # Return 0 for failed probes - will be averaged out
        return 0.0
    
    # Filter out non-positive eigenvalues (ghost eigenvalues from numerical instability)
    # instead of clipping to eps, which heavily penalizes the logdet estimate with large negative logs.
    valid = eigvals > 0
    if not np.any(valid):
        return 0.0
        
    # The quadrature rule: sum of log(eigvals) weighted by squared first elements of eigenvectors
    return np.sum(np.log(eigvals[valid]) * (eigvecs[0, valid] ** 2))

def logdet(V_op: VLinearOperator, lanczos_steps: int, n_probes: int, n_jobs: int = -1, backend: str = 'threading', random_seed: int = 42):
    """
    Estimate log-determinant using Stochastic Lanczos Quadrature (SLQ).
    
    Computes log(det(V)) using parallelized SLQ with multiple probe vectors.
    Estimate: log(det(V)) \\approx (n/m) * sum_{i=1}^m probe_i where n is dimension
    and m is number of probes.
    
    Parameters
    ----------
    V_op : VLinearOperator
        Symmetric positive-definite linear operator V.
    lanczos_steps : int
        Number of Lanczos iterations per probe.
    n_probes : int
        Number of probe vectors m.
    n_jobs : int
        Number of parallel jobs.
    backend : str, default='threading'
        Joblib parallel backend (e.g., 'threading', 'loky').
        'threading' is recommended to avoid heavy memory copying.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    float
        Estimated log-determinant: log(det(V)).
    """
    dim = V_op.shape[0]
    
    # Mathematical Crossover for Lanczos Steps:
    # A Krylov subspace spans at most the matrix dimension. 
    # Any steps beyond 'dim' add zero information and risk numerical instability.
    lanczos_steps = min(lanczos_steps, dim)
    
    # Mathematical Crossover for Probes:
    n_probes = min(n_probes, dim)

    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(random_seed).spawn(n_probes)]
    
    # Process blocks directly using V_op.matmat (via block SLQ algorithm)
    # Memory consideration: Q_cur, Q_prev and W take N x block_size memory.
    # We slice probes into blocks (e.g., max 64 or 128 probes at once) to avoid memory issues.
    block_size = min(32, n_probes)
    logdet_est = 0.0

    if block_size < 2:
        # Fall back to single loop or Joblib
        with parallel_config(backend=backend, n_jobs=n_jobs):
            result = Parallel(return_as="generator")(delayed(slq_probe)(V_op, lanczos_steps, s) for s in seeds)
            logdet_est = sum(result)
    else:
        # Block operation is beneficial, run blocks in parallel
        with parallel_config(backend=backend, n_jobs=n_jobs):
            batches = [seeds[i:min(i + block_size, n_probes)] for i in range(0, n_probes, block_size)]
            result = Parallel(return_as="generator")(
                delayed(slq_probes_block)(V_op, lanczos_steps, batch) for batch in batches
            )
            logdet_est = sum(result)

    return dim * logdet_est / n_probes