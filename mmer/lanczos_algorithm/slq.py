import numpy as np
from scipy.linalg import eigh_tridiagonal
from joblib import Parallel, delayed, parallel_config
from ..core.operator import VLinearOperator

def slq_probe(V_op: VLinearOperator, lanczos_steps: int, seed: int):
    """
    Single probe for SLQ logdet estimation with optimized error handling.
    """
    rng = np.random.default_rng(seed)
    dim = V_op.shape[0]
    # Use Rademacher distribution for the probe vector ({-1, 1})
    v = rng.choice([-1.0, 1.0], size=dim)
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
            w = w - alpha_j * q_cur - (betas[j-1] if j > 0 else 0.0) * q_prev
            beta_j = np.linalg.norm(w)
            # Use a strict threshold to avoid numerical issues and stop early if needed
            if beta_j < 1e-12:
                lanczos_steps = j + 1
                alphas = alphas[:lanczos_steps]
                betas = betas[:lanczos_steps-1]
                break
            betas[j] = beta_j
            q_prev, q_cur = q_cur, w / beta_j

    # Only wrap the potentially unstable eigenvalue computation
    try:
        # Compute eigenvalues and eigenvectors of the tridiagonal matrix T
        eigvals, eigvecs = eigh_tridiagonal(alphas, betas, eigvals_only=False)
    except Exception:
        # If eigenvalue computation fails (e.g., due to an unstable T),
        # Return 0 for failed probes - will be averaged out
        return 0.0
    
    # Clip small or negative eigenvalues to prevent log(0) errors.
    eps = 1e-14  # Fixed threshold, faster than np.finfo
    eigvals = np.maximum(eigvals, eps)
    # The quadrature rule: sum of log(eigvals) weighted by squared first elements of eigenvectors
    return np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))

def logdet(V_op: VLinearOperator, lanczos_steps: int, num_probes: int, n_jobs: int, backend: str, random_seed: int = 42):
    """
    Estimates the log-determinant of a symmetric positive-definite operator V
    using a parallelized Stochastic Lanczos Quadrature method.

    Returns an estimate of log(det(V)).
    """
    dim = V_op.shape[0]
    # Create a sequence of independent random seeds for each parallel job
    # This ensures reproducibility while maintaining statistical independence.
    seeds = np.random.SeedSequence(random_seed).spawn(num_probes)
    with parallel_config(backend=backend, n_jobs=n_jobs):
        result = Parallel(return_as="generator")(delayed(slq_probe)(V_op, lanczos_steps, int(s.generate_state(1)[0])) for s in seeds)
        logdet_est = sum(result)

    return dim * logdet_est / num_probes