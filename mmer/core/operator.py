import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from joblib import Parallel, delayed, parallel_config
from scipy.sparse.linalg import cg
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh
from scipy.sparse.linalg import spsolve
from .residual import Residual
from .random_effect import RandomEffect

class VLinearOperator(LinearOperator):
    """
    A linear operator that represents the marginal covariance matrix V and its matrix-vector product.
    V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ + R
    """
    def __init__(self, random_effects: tuple[RandomEffect], residual: Residual):
        self.random_effects = random_effects
        self.residual = residual
        super().__init__(dtype=np.float64, shape=(self.residual.m * self.residual.n, self.residual.m * self.residual.n))

    def _matvec(self, x_vec: np.ndarray):
        """
        Computes the marginal covariance matrix-vector product V @ x_vec,
        where V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ + R.
        returns:
            1d array (Mn,)
        """
        Vx = self.residual.full_cov_matvec(x_vec)
        for re in self.random_effects:
            Vx += re.full_cov_matvec(x_vec)
        return Vx

    def _adjoint(self):
        """Implements the adjoint operator Vᵀ. Since V is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.random_effects, self.residual))

class ResidualPreconditioner(LinearOperator):
    """
    The Lightweight Preconditioner: P⁻¹ = R⁻¹ = φ⁻¹ ⊗ Iₙ

    This is the simplest and cheapest preconditioner. It approximates the full
    covariance V with only its residual component R, ignoring all random effects.
    """
    def __init__(self, resid_cov_inv: np.ndarray, n: int, m: int):
        self.cov_inv = resid_cov_inv
        self.n = n
        self.m = m
        super().__init__(dtype=np.float64, shape=(self.m * self.n, self.m * self.n))

    def _matvec(self, x_vec: np.ndarray):
        """
        Computes (φ⁻¹ ⊗ Iₙ) @ x_vec.
        knowing that (XᵀC)ᵀ = CᵀX = CX
        """
        Px = self.cov_inv @ x_vec.reshape((self.m, self.n))
        return Px.ravel()
    
    def _adjoint(self):
        """ Since P is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.cov_inv, self.n, self.m))
    
# ====================== Main Adaptive Correction Function ======================

def compute_cov_correction(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, method: str, n_jobs: int, backend: str):
    """
    Adaptively computes the uncertainty correction matrices T and W.
    The method can be 'detr' for deterministic, 'bste' for block stochastic trace estimation,
    or 'ste' for stochastic trace estimation
    """
    re = V_op.random_effects[k]
    block_size = re.q * re.o
    n_probes = min(max(20, block_size // 10), 100)
    if method == 'detr':
        # For small problems, the exact method is better and often faster.
        return compute_cov_correction_detr(k, V_op, M_op, n_jobs, backend)
    elif method == 'bste':
        # For medium-sized to large problems, the stochastic block trace method is necessary for performance.
        return compute_cov_correction_bste(k, V_op, M_op, n_probes, n_jobs, backend)
    elif method == 'ste':
        # For very large problems, the stochastic trace method is efficient.
        return compute_cov_correction_ste(k, V_op, M_op, n_probes, n_jobs, backend)
    elif method == 'woodbury': # Add this new option
        return compute_cov_correction_woodbury(k, V_op, n_jobs, backend)

def _generate_rademacher_probes(n_rows, n_probes, seed=42):
    """Generates random probe vectors from a Rademacher distribution ({-1, 1})."""
    rng = np.random.default_rng(seed)
    probes = rng.integers(0, 2, size=(n_rows, n_probes), dtype=np.int8)
    probes *= 2
    probes -= 1
    return probes.astype(np.float64)

# ====================== Stochastic Trace Estimation for Correction ======================

def compute_cov_correction_ste(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, n_probes: int, n_jobs: int, backend: str):
    """
    Computes the uncertainty correction matrices T: (m, m) and W: (m * q, m * q)
    using the Hutchinson Stochastic Trace Estimation method.
    It only computes diagonal elements, ignoring off-diagonal elements in T and W (no issue for m=q=1).
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
    Computes the product of probe vector and matrix vector product P(C @ P), where C=D(Iₘ ⊗ Z)ᵀ V⁻¹(Iₘ ⊗ Z)D.
    """
    seed = 42 + probe_index
    re = V_op.random_effects[k]
    probe_vector = _generate_rademacher_probes(re.m * re.q * re.o, 1, seed).ravel()
    v1 = re.kronZ_D_matvec(probe_vector)
    v2, info = cg(A=V_op, b=v1, M=M_op)
    if info != 0:
        print(f"Warning: CG solver (V⁻¹(Iₘ ⊗ Z)D) did not converge. Info={info}")
    probe_vector *= re.kronZ_D_T_matvec(v2)
    return probe_vector

# ====================== Block Stochastic Trace Estimation for Correction ======================

def compute_cov_correction_bste(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, n_probes: int, n_jobs: int, backend: str):
    """
    Computes the uncertainty correction matrices T: (m, m) and W: (m * q, m * q)
    using the Hutchinson Stochastic Trace Estimation method.
    It computes diagonal and off-diagonal elements of T and W.
    However, only diagonals of each response block are considered (No issue for q=1).
    It uses the symmetry of the covariance matrix to reduce computations.
    """
    m = V_op.residual.m
    q = V_op.random_effects[k].q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(delayed(cov_correction_per_response_bste)(n_probes, k, V_op, M_op, col) for col in range(m))

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

def cov_correction_per_response_bste(n_probes: int, k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, col: int):
    """
    Stochastic Block Trace Estimation method for Cᵢⱼ=Dᵢ(Iₘ ⊗ Z)ᵀ V⁻¹(Iₘ ⊗ Z)Dⱼ for i >= j.
    """
    m = V_op.residual.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col

    diag_C = np.zeros(num_blocks * block_size)
    vec_cg = np.zeros(V_op.shape[0])
    vec = np.zeros(m * block_size)
    probe_vector = np.zeros(block_size)
    for i in range(n_probes):
        seed = 42 + i
        probe_vector[...] = _generate_rademacher_probes(block_size, 1, seed).ravel()
        vec[col * block_size : (col + 1) * block_size] = probe_vector
        vec_cg[...] = re.kronZ_D_matvec(vec)
        vec_cg[...], info = cg(A=V_op, b=vec_cg, M=M_op)
        if info != 0:
            print(f"Warning: CG solver (V⁻¹(Iₘ ⊗ Z)D) did not converge. Info={info}")
        lower_c = re.kronZ_D_T_matvec(vec_cg)[col * block_size:]
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

def compute_cov_correction_detr(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, n_jobs: int, backend: str):
    """
    Computes the uncertainty correction matrices T: (m, m) and W: (m * q, m * q)
    using the deterministic method.
    It computes full diagonal and off-diagonal elements of T and W.
    It uses the symmetry of the covariance matrix to reduce computations.
    """
    m = V_op.residual.m
    q = V_op.random_effects[k].q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with parallel_config(backend=backend, n_jobs=n_jobs):
        results = Parallel(return_as="generator")(delayed(cov_correction_per_response)(k, V_op, M_op, col) for col in range(m))

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

def cov_correction_per_response(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, col: int):
    """
    Computes the element of the uncertainty correction matrix T that is:
        Tᵢⱼ = trace((Zₖᵀ Zₖ) Σᵢⱼ)
    using the random effect conditional covariance
        Σ = D - D (Iₘ ⊗ Z)ᵀ V⁻¹ (Iₘ ⊗ Z) D
    as well as the correction matrix W per response.
    """
    m = V_op.residual.m
    re = V_op.random_effects[k]
    q, o = re.q, re.o
    block_size = q * o
    num_blocks = m - col
    lower_sigma = np.empty((num_blocks * block_size, block_size))
    base_idx = col * block_size
    vec_cg = np.zeros(V_op.shape[0])
    vec = np.zeros(m * block_size)
    for i in range(block_size):
        vec[base_idx + i] = 1.0
        vec_cg[...] = re.kronZ_D_matvec(vec)
        vec_cg[...], info = cg(A=V_op, b=vec_cg, M=M_op)
        if info != 0:
            print(f"Warning: CG solver (V⁻¹(Iₘ ⊗ Z)D) did not converge. Info={info}")
        lower_sigma[:, i] = (re.D_matvec(vec) - re.kronZ_D_T_matvec(vec_cg))[col * block_size:]
        vec[base_idx + i] = 0

    sigma_blocks = lower_sigma.reshape(num_blocks, block_size, block_size)
    elementwise_prod = re.ZTZ.multiply(sigma_blocks)
    T_traces = elementwise_prod.sum(axis=(1, 2))
    W_blocks = lower_sigma.reshape(num_blocks, q, o, q, o).sum(axis=(2, 4))

    return col, T_traces, W_blocks

# ====================== Woodbury for Correction ======================

def compute_cov_correction_woodbury(k: int, V_op: VLinearOperator, n_jobs: int, backend: str):
    """
    Computes T and W using the Woodbury identity, implemented by building the
    required sparse matrices and using a robust sparse linear solver. This approach
    is mathematically sound and avoids the flawed assumptions of previous versions.
    """
    # === Step 1: Extract Model Components ===
    target_re = V_op.random_effects[k]
    residual = V_op.residual
    m, q_k, o_k = residual.m, target_re.q, target_re.o
    M_q_o = m * q_k * o_k

    phi_inv = np.linalg.inv(residual.cov)

    tau_k = target_re.cov
    L_tau_k = cholesky(tau_k, lower=True)

    # === Step 2: Build the Large Sparse Matrices Correctly ===

    # D_k = τ_k ⊗ I_{o_k}
    D_k_sparse = sparse.kron(tau_k, sparse.eye(o_k), format='csc')

    # D_k^½ = τ_k^½ ⊗ I_{o_k}
    D_k_half_sparse = sparse.kron(L_tau_k, sparse.eye(o_k), format='csc')

    # Z_kᵀZ_k
    ZkTZk_sparse = target_re.ZTZ.tocsc()

    # The middle term: ϕ⁻¹ ⊗ (Z_kᵀZ_k)
    middle_term_sparse = sparse.kron(phi_inv, ZkTZk_sparse, format='csc')

    # Assemble A_kk = (D_k^½)ᵀ [ ϕ⁻¹ ⊗ (Z_kᵀZ_k) ] D_k^½
    # This involves sparse matrix multiplications.
    term1 = middle_term_sparse @ D_k_half_sparse
    A_kk_sparse = D_k_half_sparse.T @ term1

    # === Step 3: Solve the Sparse Linear System ===
    # We need to solve (I + A_kk) X = B for the correction term C_k.
    # The full conditional covariance is Σ_k = D_k - C_k, where C_k = D_k (I+A_kk)⁻¹ A_kk.
    # It's more direct to solve for Σ_k = D_k (I + A_kk)⁻¹.
    
    I_plus_A_kk = sparse.eye(M_q_o, format='csc') + A_kk_sparse

    # We solve (I + A_kk) Σ_k = D_k using a sparse solver.
    # This gives us Σ_k = (I + A_kk)⁻¹ D_k.
    # The RHS D_k is sparse, so we can solve efficiently.
    Sigma_k_sparse = spsolve(I_plus_A_kk, D_k_sparse, use_umfpack=True)
    Sigma_k_sparse = sparse.csc_matrix(Sigma_k_sparse)

    # === Step 4: Calculate W_k and T_k from Σ_k ===

    # W_k is the sum of the (m*q_k, m*q_k) diagonal blocks of Σ_k
    W_k = np.zeros_like(tau_k)
    for i in range(o_k):
        start = i * m * q_k
        end = (i + 1) * m * q_k
        W_k += Sigma_k_sparse[start:end, start:end].toarray()

    # T_k requires the trace of products with Σ_k's blocks.
    T_k = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m): # Exploit symmetry
            # We need the (i,j)-th block of Σ_k, which is a (q_k*o_k, q_k*o_k) matrix.
            # This requires careful slicing.
            rows = np.array([row for r_idx in range(o_k) for row in range(r_idx*m*q_k + i*q_k, r_idx*m*q_k + (i+1)*q_k)])
            cols = np.array([col for c_idx in range(o_k) for col in range(c_idx*m*q_k + j*q_k, c_idx*m*q_k + (j+1)*q_k)])
            
            sigma_ij_block = Sigma_k_sparse[rows, :][:, cols]
            
            # T_ij = trace(ZkTZk * sigma_ji)
            # Note the index swap to (j,i) due to linear algebra conventions.
            if i == j:
                 trace_val = ZkTZk_sparse.multiply(sigma_ij_block).sum()
            else:
                 # For off-diagonal, need to get sigma_ji_block
                 rows_ji = np.array([row for r_idx in range(o_k) for row in range(r_idx*m*q_k + j*q_k, r_idx*m*q_k + (j+1)*q_k)])
                 cols_ji = np.array([col for c_idx in range(o_k) for col in range(c_idx*m*q_k + i*q_k, c_idx*m*q_k + (i+1)*q_k)])
                 sigma_ji_block = Sigma_k_sparse[rows_ji, :][:, cols_ji]
                 trace_val = ZkTZk_sparse.multiply(sigma_ji_block).sum()

            T_k[i, j] = trace_val
            if i != j:
                T_k[j, i] = trace_val

    return T_k, W_k