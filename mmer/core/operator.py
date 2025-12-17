import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from joblib import Parallel, delayed, parallel_config
from .residual import Residual
from .random_effect import RandomEffect

class VLinearOperator(LinearOperator):
    """
    A linear operator that represents the marginal covariance matrix V and its matrix-vector product.
    V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ + R
    """
    def __init__(self, random_effects: tuple[RandomEffect], residual: Residual):
        """
        Initialize the VLinearOperator.

        Parameters
        ----------
        random_effects : tuple[RandomEffect]
            Tuple of random effect components.
        residual : Residual
            Residual covariance component.
        """
        self.random_effects = random_effects
        self.residual = residual
        super().__init__(dtype=np.float64, shape=(self.residual.m * self.residual.n, self.residual.m * self.residual.n))

    def _matvec(self, x_vec: np.ndarray):
        """
        Compute the marginal covariance matrix-vector product V @ x_vec.
        
        V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ + R

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector of shape (Mn,).

        Returns
        -------
        np.ndarray
            Result of V @ x_vec, shape (Mn,).
        """
        Vx = self.residual._full_cov_matvec(x_vec)
        for re in self.random_effects:
            Vx += re._full_cov_matvec(x_vec)
        return Vx

    def _adjoint(self):
        """
        Implement the adjoint operator Vᵀ.

        Returns
        -------
        VLinearOperator
            Self, since V is symmetric.
        """
        return self
    
    def __reduce__(self):
        """
        Enable pickling for multiprocessing.

        Returns
        -------
        tuple
            Class and arguments for unpickling.
        """
        return (self.__class__, (self.random_effects, self.residual))

class ResidualPreconditioner(LinearOperator):
    """
    The Lightweight Preconditioner: P⁻¹ = R⁻¹ = φ⁻¹ ⊗ Iₙ.

    This is the simplest and cheapest preconditioner. It approximates the full
    covariance V with only its residual component R, ignoring all random effects.

    Parameters
    ----------
    resid_cov_inv : np.ndarray
        Inverse of residual covariance matrix φ⁻¹.
    n : int
        Number of observations.
    m : int
        Number of responses.
    """
    def __init__(self, resid_cov_inv: np.ndarray, n: int, m: int):
        self.cov_inv = resid_cov_inv
        self.n = n
        self.m = m
        super().__init__(dtype=np.float64, shape=(self.m * self.n, self.m * self.n))

    def _matvec(self, x_vec: np.ndarray):
        """
        Compute (φ⁻¹ ⊗ Iₙ) @ x_vec.

        Parameters
        ----------
        x_vec : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Result of preconditioner applied to x_vec.
        """
        Px = self.cov_inv @ x_vec.reshape((self.m, self.n))
        return Px.ravel()
    
    def _adjoint(self):
        """
        Implement the adjoint operator.

        Returns
        -------
        ResidualPreconditioner
            Self, since P is symmetric.
        """
        return self
    
    def __reduce__(self):
        """
        Enable pickling for multiprocessing.

        Returns
        -------
        tuple
            Class and arguments for unpickling.
        """
        return (self.__class__, (self.cov_inv, self.n, self.m))
    
# ====================== Main Adaptive Correction Function ======================

def compute_cov_correction(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, method: str, n_jobs: int, backend: str):
    """
    Adaptively compute the uncertainty correction matrices T and W.

    Parameters
    ----------
    k : int
        Index of the grouping factor for the random effects.
    V_op : VLinearOperator
        Marginal covariance linear operator.
    M_op : ResidualPreconditioner
        Preconditioner operator.
    method : str
        Method for covariance correction. Options: 'ste' (Stochastic Trace Estimation), 'bste' (Block Stochastic Trace Estimation), 'de' (Deterministic Estimation).
    n_jobs : int
        Number of parallel jobs.
    backend : str
        Joblib backend for parallelization.

    Returns
    -------
    tuple
        Covariance correction matrices (T, W).
    """
    re = V_op.random_effects[k]
    block_size = re.q * re.o
    n_probes = min(max(20, block_size // 10), 100)
    if method == 'de':
        # For small problems, the exact method is better and often faster.
        return compute_cov_correction_de(k, V_op, M_op, n_jobs, backend)
    elif method == 'bste':
        # For medium-sized to large problems, the stochastic block trace method is necessary for performance.
        return compute_cov_correction_bste(k, V_op, M_op, n_probes, n_jobs, backend)
    elif method == 'ste':
        # For very large problems, the stochastic trace method is efficient.
        return compute_cov_correction_ste(k, V_op, M_op, n_probes, n_jobs, backend)

def _generate_rademacher_probes(n_rows, n_probes, seed=42):
    """
    Generate random probe vectors from a Rademacher distribution ({-1, 1}).

    Parameters
    ----------
    n_rows : int
        Number of rows in each probe vector.
    n_probes : int
        Number of probe vectors to generate.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    np.ndarray
        Array of shape (n_rows, n_probes) with values in {-1, 1}.
    """
    rng = np.random.default_rng(seed)
    probes = rng.integers(0, 2, size=(n_rows, n_probes), dtype=np.int8)
    probes *= 2
    probes -= 1
    return probes.astype(np.float64)

# ====================== Stochastic Trace Estimation for Correction ======================

def compute_cov_correction_ste(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, n_probes: int, n_jobs: int, backend: str):
    """
    Compute uncertainty correction matrices T and W using Hutchinson STE.

    Computes diagonal elements only, ignoring off-diagonal elements in T and W.

    Parameters
    ----------
    k : int
        Index of the random effect.
    V_op : VLinearOperator
        Marginal covariance linear operator.
    M_op : ResidualPreconditioner
        Preconditioner operator.
    n_probes : int
        Number of probe vectors for stochastic estimation.
    n_jobs : int
        Number of parallel jobs.
    backend : str
        Joblib backend for parallelization.

    Returns
    -------
    tuple
        Diagonal correction matrices (T: (m, m), W: (m*q, m*q)).
    """
    m = V_op.residual.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    diag_C = np.zeros(m * q * o)
    with parallel_config(backend=backend, n_jobs=n_jobs):
        pcp_gen = Parallel(return_as="generator")(delayed(_compute_C_probe)(i, k, V_op, M_op) for i in range(n_probes))
        for diag_pcp in pcp_gen:
            diag_C += diag_pcp

    diag_C /= n_probes
    diag_sigma = np.repeat(np.diag(re.cov), o)
    diag_sigma -= diag_C

    W_diag = diag_sigma.reshape((m, q, o)).sum(axis=2).ravel()
    sigma_mat  = diag_sigma.reshape(m, q * o)
    ZTZ_diag = re.ZTZ.diagonal()
    T_diag = sigma_mat.dot(ZTZ_diag)

    return np.diag(T_diag), np.diag(W_diag)

def _compute_C_probe(probe_index: int, k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner):
    """
    Compute the probe-based C matrix element for C = D(Iₘ ⊗ Z)ᵀ V⁻¹(Iₘ ⊗ Z)D.

    Parameters
    ----------
    probe_index : int
        Index of the probe vector.
    k : int
        Index of the random effect.
    V_op : VLinearOperator
        Marginal covariance linear operator.
    M_op : ResidualPreconditioner
        Preconditioner operator.

    Returns
    -------
    np.ndarray
        Element-wise product of probe vector and matrix-vector product.
    """
    seed = 42 + probe_index
    re = V_op.random_effects[k]
    probe_vector = _generate_rademacher_probes(re.m * re.q * re.o, 1, seed).ravel()
    v1 = re._kronZ_D_matvec(probe_vector)
    v2, info = cg(A=V_op, b=v1, M=M_op)
    if info != 0:
        print(f"Warning: CG solver (V⁻¹(Iₘ ⊗ Z)D) did not converge. Info={info}")
    probe_vector *= re._kronZ_D_T_matvec(v2)
    return probe_vector

# ====================== Block Stochastic Trace Estimation for Correction ======================

def compute_cov_correction_bste(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, n_probes: int, n_jobs: int, backend: str):
    """
    Compute uncertainty correction matrices T and W using block STE.

    Computes diagonal and off-diagonal elements of T and W. Only diagonals of each
    response block are considered. Uses symmetry to reduce computations.

    Parameters
    ----------
    k : int
        Index of the random effect.
    V_op : VLinearOperator
        Marginal covariance linear operator.
    M_op : ResidualPreconditioner
        Preconditioner operator.
    n_probes : int
        Number of probe vectors for stochastic estimation.
    n_jobs : int
        Number of parallel jobs.
    backend : str
        Joblib backend for parallelization.

    Returns
    -------
    tuple
        Correction matrices (T: (m, m), W: (m*q, m*q)).
    """
    m = V_op.residual.m
    q = V_op.random_effects[k].q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(delayed(_cov_correction_per_response_bste)(n_probes, k, V_op, M_op, col) for col in range(m))

        for col, T_lower_traces, W_lower_diags in results:
            for i, (trace, W_diag) in enumerate(zip(T_lower_traces, W_lower_diags)):
                row = col + i
                # --- Assemble T ---
                T[col, row] = T[row, col] = trace
                # --- Assemble W ---
                r_slice = slice(row * q, (row + 1) * q)
                c_slice = slice(col * q, (col + 1) * q)
                np.fill_diagonal(W[r_slice, c_slice], W_diag)
                if row != col:
                    np.fill_diagonal(W[c_slice, r_slice], W_diag)

    return T, W

def _cov_correction_per_response_bste(n_probes: int, k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, col: int):
    """
    Compute block STE for Σ = D - D(Iₘ ⊗ Z)ᵀ V⁻¹(Iₘ ⊗ Z)D.

    Computes Cᵢⱼ = Dᵢ(Iₘ ⊗ Z)ᵀ V⁻¹(Iₘ ⊗ Z)Dⱼ for i >= j.

    Parameters
    ----------
    n_probes : int
        Number of probe vectors for stochastic estimation.
    k : int
        Index of the random effect.
    V_op : VLinearOperator
        Marginal covariance linear operator.
    M_op : ResidualPreconditioner
        Preconditioner operator.
    col : int
        Column index for block processing.

    Returns
    -------
    tuple
        (col, T_traces, W_diag_blocks) for assembly.
    """
    m = V_op.residual.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col

    diag_C = np.zeros(num_blocks * block_size)
    vec = np.zeros(m * block_size)
    for i in range(n_probes):
        seed = 42 + i
        probe_vector = _generate_rademacher_probes(block_size, 1, seed).ravel()
        vec[col * block_size : (col + 1) * block_size] = probe_vector
        vec_cg = re._kronZ_D_matvec(vec)
        vec_cg, info = cg(A=V_op, b=vec_cg, M=M_op)
        if info != 0:
            print(f"Warning: CG solver (V⁻¹(Iₘ ⊗ Z)D) did not converge. Info={info}")
        lower_c = re._kronZ_D_T_matvec(vec_cg)[col * block_size:]
        vec[col * block_size : (col + 1) * block_size] = 0
        diag_C += (lower_c.reshape(num_blocks, block_size) * probe_vector).ravel()

    diag_C /= n_probes
    sub_cov = re.cov[col*q:, col*q:(col+1)*q]
    diags = sub_cov.reshape(num_blocks, q, q).diagonal(axis1=1, axis2=2)
    diag_sigma = np.repeat(diags, o, axis=1).ravel()
    diag_sigma -= diag_C

    ZTZ_diag = re.ZTZ.diagonal()
    sigma_mat  = diag_sigma.reshape(num_blocks, q * o)
    T_traces = sigma_mat.dot(ZTZ_diag)
    W_diag_blocks = diag_sigma.reshape(num_blocks, q, o).sum(axis=2)

    return col, T_traces, W_diag_blocks

# ====================== Deterministic for Correction ======================

def compute_cov_correction_de(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, n_jobs: int, backend: str):
    """
    Compute uncertainty correction matrices T and W using deterministic method.

    Computes full diagonal and off-diagonal elements of T and W.
    Uses symmetry to reduce computations.

    Parameters
    ----------
    k : int
        Index of the random effect.
    V_op : VLinearOperator
        Marginal covariance linear operator.
    M_op : ResidualPreconditioner
        Preconditioner operator.
    n_jobs : int
        Number of parallel jobs.
    backend : str
        Joblib backend for parallelization.

    Returns
    -------
    tuple
        Correction matrices (T: (m, m), W: (m*q, m*q)).
    """
    m = V_op.residual.m
    q = V_op.random_effects[k].q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(delayed(_cov_correction_per_response_de)(k, V_op, M_op, col) for col in range(m))

        for col, T_lower_traces, W_lower_blocks in results:
            for i, (trace, W_block) in enumerate(zip(T_lower_traces, W_lower_blocks)):
                row = col + i
                # --- Assemble T ---
                T[col, row] = T[row, col] = trace
                # --- Assemble W ---
                r_slice = slice(row * q, (row + 1) * q)
                c_slice = slice(col * q, (col + 1) * q)
                W[r_slice, c_slice] = W_block
                if row != col:
                    W[c_slice, r_slice] = W_block.T

    return T, W

def _cov_correction_per_response_de(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, col: int):
    """
    Compute uncertainty correction elements Tᵢⱼ = trace((Zₖᵀ Zₖ) Σᵢⱼ).

    Uses conditional covariance Σ = D - D(Iₘ ⊗ Z)ᵀ V⁻¹(Iₘ ⊗ Z)D and
    computes correction matrix W per response.

    Parameters
    ----------
    k : int
        Index of the random effect.
    V_op : VLinearOperator
        Marginal covariance linear operator.
    M_op : ResidualPreconditioner
        Preconditioner operator.
    col : int
        Column index for processing.

    Returns
    -------
    tuple
        (col, T_traces, W_blocks) for assembly.
    """
    m = V_op.residual.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col
    lower_sigma = np.empty((num_blocks * block_size, block_size))
    base_idx = col * block_size
    vec = np.zeros(m * block_size)
    for i in range(block_size):
        vec[base_idx + i] = 1.0
        vec_cg = re._kronZ_D_matvec(vec)
        vec_cg, info = cg(A=V_op, b=vec_cg, M=M_op)
        if info != 0:
            print(f"Warning: CG solver (V⁻¹(Iₘ ⊗ Z)D) did not converge. Info={info}")
        lower_sigma[:, i] = (re._D_matvec(vec) - re._kronZ_D_T_matvec(vec_cg))[col * block_size:]
        vec[base_idx + i] = 0

    sigma_blocks = lower_sigma.reshape(num_blocks, block_size, block_size)
    elementwise_prod = re.ZTZ.multiply(sigma_blocks)
    T_traces = elementwise_prod.sum(axis=(1, 2))
    W_blocks = lower_sigma.reshape(num_blocks, q, o, q, o).sum(axis=(2, 4))

    return col, T_traces, W_blocks