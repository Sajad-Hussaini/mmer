import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigh_tridiagonal
import warnings
from joblib import Parallel, delayed, cpu_count

NJOBS = max(1, int(cpu_count() * 0.70))

def slq_logdet2(V_op: LinearOperator, lanczos_steps: int = 50, num_probes: int = 30) -> float:
    """
    Estimates the log-determinant of a symmetric positive-definite operator V.

    This function uses the Stochastic Lanczos Quadrature (SLQ) method, which
    combines Hutchinson's trace estimator with Lanczos quadrature to provide a
    scalable, matrix-free estimate of log(det(V)).

    Parameters
    ----------
    V_op : scipy.sparse.linalg.LinearOperator
        The linear operator for the matrix V. It must represent a symmetric
        positive-definite matrix.
    lanczos_steps : int, optional
        The number of Lanczos iterations to perform for each probe vector.
        This controls the accuracy of the quadrature rule. Higher is more
        accurate but more computationally expensive. Default is 50.
    num_probes : int, optional
        The number of random probe vectors to use for the stochastic trace
        estimation. Higher is more accurate but more computationally expensive.
        Default is 30.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns
    -------
    float
        An estimate of the log-determinant of V.
    """
    dim = V_op.shape[0]
    logdet_est = 0.0
    rng = np.random.default_rng(seed=33)
    
    for _ in range(num_probes):
        # Use Rademacher distribution for the probe vector
        v = rng.choice([-1.0, 1.0], size=dim)

        # The core Lanczos iteration
        # This builds the tridiagonal matrix T implicitly
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
                if beta_j < 1e-10:
                    # Krylov subspace is exhausted, truncate and exit loop
                    lanczos_steps = j + 1
                    alphas = alphas[:lanczos_steps]
                    betas = betas[:lanczos_steps-1]
                    break
                betas[j] = beta_j
                q_prev, q_cur = q_cur, w / beta_j
    
        # Step 2: Use Gaussian quadrature with eigenvalues/vectors of T
        # This is faster and more memory efficient than forming the dense T matrix
        eigvals, eigvecs = eigh_tridiagonal(alphas, betas, eigvals_only=False)

        # Step 3: Apply the log function and sum with quadrature weights
        # The weights are the squared first elements of the eigenvectors of T
        # Clip eigenvalues at a small epsilon to prevent log(0) or log(<0)
        # due to floating point inaccuracies.
        eps = np.finfo(eigvals.dtype).eps
        if np.any(eigvals <= eps):
            warnings.warn("Lanczos-generated eigenvalues are non-positive. "
                          "Clipping at machine epsilon for log calculation. "
                          "This may indicate the operator is not SPD or "
                          "lanczos_steps is too small.")
            eigvals = np.maximum(eigvals, eps)
            
        logdet_est += np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))

    # The formula for Hutchinson's trace estimator with Rademacher vectors is
    # Tr(log(V)) ≈ sum(v_i^T log(V) v_i) / num_probes.
    # The quadrature term logdet_est approximates v^T log(V) v, but for a
    # normalized vector u = v / ||v||. Since ||v||^2 = dim, we must scale
    # the result. The full expression is dim * E[u^T log(V) u].
    return dim * logdet_est / num_probes

def slq_probe(V_op, lanczos_steps, seed):
    """
    Single probe for SLQ logdet estimation.
    """
    try:
        rng = np.random.default_rng(seed)
        dim = V_op.shape[0]
        v = rng.choice([-1.0, 1.0], size=dim)
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
                if beta_j < 1e-10:
                    # Krylov subspace is exhausted, truncate and exit loop
                    lanczos_steps = j + 1
                    alphas = alphas[:lanczos_steps]
                    betas = betas[:lanczos_steps-1]
                    break
                betas[j] = beta_j
                q_prev, q_cur = q_cur, w / beta_j

        eigvals, eigvecs = eigh_tridiagonal(alphas, betas, eigvals_only=False)
        eps = np.finfo(eigvals.dtype).eps
        if np.any(eigvals <= eps):
            warnings.warn("Lanczos-generated eigenvalues are non-positive. "
                            "Clipping at machine epsilon for log calculation. "
                            "This may indicate the operator is not SPD or "
                            "lanczos_steps is too small.")
            eigvals = np.maximum(eigvals, eps)

        return np.sum(np.log(eigvals) * (eigvecs[0, :] ** 2))
    except np.linalg.LinAlgError as e:
        # warnings.warn(f"Probe failed due to LinAlgError: {e}. Skipping.")
        return 0.0

def slq_logdet(V_op: LinearOperator, lanczos_steps: int = 50, num_probes: int = 30):
    """
    Parallel SLQ logdet estimation.
    """
    dim = V_op.shape[0]
    seeds = np.random.SeedSequence(42).spawn(num_probes)
    results = Parallel(n_jobs=NJOBS, backend='loky')(delayed(slq_probe)
                                                     (V_op, lanczos_steps, int(s.generate_state(1)[0])) for s in seeds)
    logdet_est = np.sum(results)
    return dim * logdet_est / num_probes
