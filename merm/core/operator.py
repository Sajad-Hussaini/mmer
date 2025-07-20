import gc
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve
from joblib import Parallel, delayed
from scipy.sparse.linalg import cg
from .residual import Residual
from .random_effect import RandomEffect

class VLinearOperator(LinearOperator):
    """
    A linear operator that represents the marginal covariance matrix V.
    V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)ᵀ + R
    """
    def __init__(self, random_effects: dict[int, RandomEffect], residual: Residual):
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
        for re in self.random_effects.values():
            np.add(Vx, re.full_cov_matvec(x_vec), out=Vx)
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
    def __init__(self, residual: Residual | np.ndarray, n=None, m=None):
        if isinstance(residual, np.ndarray):
            self.n = n
            self.m = m
            self.cov_inv = residual
        else:
            self.n = residual.n
            self.m = residual.m
            self.cov_inv = solve(residual.cov, np.eye(self.m), assume_a='pos')

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
    
# ====================== Other Standalone Methods ======================

def compute_cov_correction(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, n_jobs: int, backend: str):
    """
    Computes the correction to the residual covariance matrix φ and
    random effect covariance matrix τ
    by constructing the uncertainty correction matrix T: (m, m)
    and W: (m * q, m * q)

    Uses symmetry of the covariance matrix to reduce computations.
    """
    m = V_op.residual.m
    q = V_op.random_effects[k].q
    T = np.zeros((m, m))
    W = np.zeros((m * q, m * q))
    with Parallel(n_jobs=n_jobs, backend=backend, return_as="generator") as parallel:
        results = parallel(delayed(cov_correction_per_response)(k, V_op, M_op, col) for col in range(m))

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

def compute_sigma_column(i: int, base_idx: int, vec: np.ndarray, k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner,
                         col: int, block_size: int):
    """ Computes the i-th column of the conditional covariance matrix Σ for lower triangular part. """
    vec[base_idx + i] = 1.0
    rhs = V_op.random_effects[k].kronZ_D_matvec(vec)
    x_sol, _ = cg(V_op, rhs, M=M_op)
    lower_sigma = (V_op.random_effects[k].D_matvec(vec) -
              V_op.random_effects[k].kronZ_D_T_matvec(x_sol))[col * block_size:]
    vec[base_idx + i] = 0.0
    return lower_sigma

def compute_T_traces(k: int, V_op: VLinearOperator, lower_sigma: np.ndarray, num_blocks: int, block_size: int):
    """
    computes the traces of the blocks of the uncertainty correction matrix T for one response column
    Tᵢⱼ = trace((Zₖᵀ Zₖ) Σᵢⱼ)
    """
    return [V_op.random_effects[k].ZTZ.multiply(lower_sigma[i * block_size:(i + 1) * block_size, :]).sum() for i in range(num_blocks)]

def cov_correction_per_response(k: int, V_op: VLinearOperator, M_op: ResidualPreconditioner, col: int):
    """
    Computes the element of the uncertainty correction matrix T that is:
        Tᵢⱼ = trace((Zₖᵀ Zₖ) Σᵢⱼ)
    using the random effect conditional covariance
        Σ = D - D (Iₘ ⊗ Z)ᵀ V⁻¹ (Iₘ ⊗ Z) D
    as well as the correction matrix W per response.
    """
    m = V_op.residual.m
    q = V_op.random_effects[k].q
    o = V_op.random_effects[k].o
    block_size = q * o
    num_blocks = m - col
    lower_sigma = np.zeros((num_blocks * block_size, block_size))
    base_idx = col * block_size
    vec = np.zeros(m * block_size)
    for i in range(block_size):
        lower_sigma[:, i] = compute_sigma_column(i, base_idx, vec, k, V_op, M_op, col, block_size)

    T_traces = compute_T_traces(k, V_op, lower_sigma, num_blocks, block_size)
    W_lower_blocks = lower_sigma.reshape(num_blocks, q, o, q, o).sum(axis=(2, 4))
    gc.collect() 
    return col, T_traces, W_lower_blocks
