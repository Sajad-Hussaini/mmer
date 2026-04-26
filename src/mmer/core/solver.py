import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu
from scipy.linalg import cho_factor, cho_solve, LinAlgError, pinvh
from ..lanczos_algorithm import slq
from .operator import VLinearOperator, ResidualPreconditioner
from .terms import RealizedRandomEffect, RealizedResidual


def _get_chol_or_inv(mat: np.ndarray) -> tuple:
    """
    Get Cholesky factor or fallback to pseudo-inverse if not positive-definite.
    """
    try:
        return cho_factor(mat, lower=True, check_finite=False), None
    except LinAlgError:
        return None, pinvh(mat)


def _invert_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Invert a symmetric positive-definite matrix.

    Uses Cholesky factorisation (``cho_factor`` / ``cho_solve``) which avoids
    allocating an explicit identity matrix and is faster than the generic
    ``solve(b=np.eye(...))`` path.  Falls back to ``pinvh`` (eigenvalue-
    thresholded pseudo-inverse) if the matrix is not positive-definite.
    """
    chol, inv = _get_chol_or_inv(mat)
    if chol is not None:
        return cho_solve(chol, np.eye(mat.shape[0]), check_finite=False)
    return inv


class BaseSolver:
    """Base class for solvers."""

    def __init__(
        self,
        realized_effects: tuple[RealizedRandomEffect, ...],
        realized_residual: RealizedResidual,
    ):
        self.realized_effects = realized_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m
        self.V_op = VLinearOperator(self.realized_effects, self.realized_residual)

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def logdet(self, **kwargs) -> float:
        """Compute or estimate log det(V)."""
        raise NotImplementedError


class IterativeSolver(BaseSolver):
    """Iterative Conjugate Gradient Solver."""

    def __init__(
        self,
        realized_effects: tuple[RealizedRandomEffect, ...],
        realized_residual: RealizedResidual,
        preconditioner: bool = True,
        cg_maxiter: int = 1000,
    ):
        super().__init__(realized_effects, realized_residual)
        self.use_preconditioner = preconditioner
        self.cg_maxiter = cg_maxiter

        self.M_op = None

        if self.use_preconditioner:
            R = self.realized_residual.term.cov
            # If the matrix is strictly singular, let it surface the error (consistent with WoodburySolver) and ask to disable preconditioning
            R_chol, R_inv = _get_chol_or_inv(R)
            if R_chol is not None:
                R_inv_pass = None
            else:
                R_inv_pass = R_inv
            self.M_op = ResidualPreconditioner(R_inv_pass, R_chol, self.n, self.m)

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        if marginal_residual.ndim == 2:
            prec_resid = np.empty_like(marginal_residual)
            for i, rhs in enumerate(marginal_residual.T):
                sol, info = cg(A=self.V_op, b=rhs, M=self.M_op, maxiter=self.cg_maxiter)
                if info < 0:
                    raise RuntimeError(f"Conjugate Gradient breakdown (info={info}).")
                prec_resid[:, i] = sol
        else:
            prec_resid, info = cg(
                A=self.V_op, b=marginal_residual, M=self.M_op, maxiter=self.cg_maxiter
            )
            if info < 0:
                raise RuntimeError(f"Conjugate Gradient breakdown (info={info}).")

        return prec_resid

    def logdet(
        self,
        slq_steps: int,
        n_probes: int,
        n_jobs: int = -1,
        backend: str = "threading",
    ) -> float:
        """Estimate log det(V) via Stochastic Lanczos Quadrature."""
        return slq.logdet(self.V_op, slq_steps, n_probes, n_jobs, backend)


class WoodburySolver(BaseSolver):
    """Woodbury Matrix Identity Direct Solver."""

    def __init__(
        self,
        realized_effects: tuple[RealizedRandomEffect, ...],
        realized_residual: RealizedResidual,
    ):
        super().__init__(realized_effects, realized_residual)

        R = self.realized_residual.term.cov
        # Only the dense inverse is needed; Cholesky factor is not reused.
        self.R_inv_dense = _invert_matrix(R)
        R_inv_sp = sparse.csr_array(self.R_inv_dense)

        self.is_fast_path = len(self.realized_effects) == 1

        if self.is_fast_path:
            # --- Fast Path for Single Grouping Factor (k=1) ---
            # For a single random effect term, the Woodbury S matrix is mathematically equivalent
            # to a block-diagonal matrix. This is because Z^T Z only couples observations
            # within the same group level. By exploiting this, we can explicitly compute the
            # o independent blocks of size (m*q x m*q) and solve them in parallel using NumPy's
            # batched linear algebra. This avoids the O((m*q*o)^3) bottleneck of sparse LU
            # decomposition, achieving a massive speedup and reducing memory allocations.
            re = self.realized_effects[0]
            ZTZ_coo = re.ZTZ.tocoo()
            u_idx = ZTZ_coo.row // re.o
            l_idx = ZTZ_coo.row % re.o
            v_idx = ZTZ_coo.col // re.o

            # Extract Z^T Z cross-products into A tensor of shape (o, q, q)
            A = np.zeros((re.o, re.q, re.q))
            np.add.at(A, (l_idx, u_idx, v_idx), ZTZ_coo.data)

            # Construct the S matrix blocks: S_l = R^{-1} ⊗ A_l + D^{-1}
            S_kron = np.einsum("ij,luv->liujv", self.R_inv_dense, A).reshape(
                re.o, self.m * re.q, self.m * re.q
            )
            D_inv_dense = _invert_matrix(re.term.cov)
            self.S_batch = S_kron + D_inv_dense
            self.S_factor = None
        else:
            self.S_batch = None
            self.S_chol = None
            self.S_dense_fallback = None
            # --- General Path for Multiple Grouping Factors (k>1) ---
            # Construct S matrix once.
            S_blocks = []
            for i, re_i in enumerate(self.realized_effects):
                row_blocks = []
                for j, re_j in enumerate(self.realized_effects):
                    Z_i_T_Z_j = re_i.ZTZ if i == j else re_i.Z.T @ re_j.Z
                    S_ij = sparse.kron(R_inv_sp, Z_i_T_Z_j)

                    if i == j:
                        D_inv = _invert_matrix(re_i.term.cov)
                        I_oi = sparse.eye_array(re_i.o, format='csr')
                        C_inv_ii = sparse.kron(sparse.csr_array(D_inv), I_oi)
                        S_ij = S_ij + C_inv_ii

                    row_blocks.append(S_ij)
                S_blocks.append(row_blocks)

            if S_blocks:
                S_mat = sparse.block_array(S_blocks, format='csc')
                # Factorize once and reuse solves across E/M-step calls.
                self.S_factor = splu(S_mat)
            else:
                self.S_factor = None

    def _build_v1(self, A_inv_x: np.ndarray, is_2d: bool) -> np.ndarray:
        """Compute v1 = (I_m ⊗ Z^T) A^{-1} x for all random-effect terms."""
        v1_list = [re_i._kronZ_T_matvec(A_inv_x) for re_i in self.realized_effects]
        return np.vstack(v1_list) if is_2d else np.concatenate(v1_list)

    def _build_v3(self, v2: np.ndarray, is_2d: bool, K: int) -> np.ndarray:
        """Compute v3 = (I_m ⊗ Z) v2 for all random-effect terms."""
        if is_2d:
            v3 = np.zeros((self.m * self.n, K))
        else:
            v3 = np.zeros(self.m * self.n)

        offset = 0
        for re_i in self.realized_effects:
            size_i = self.m * re_i.q * re_i.o
            v2_i = v2[offset : offset + size_i]
            v3 += re_i._kronZ_matvec(v2_i)
            offset += size_i
        return v3

    def _apply_R_inv_kron(self, x: np.ndarray) -> np.ndarray:
        r"""Computes (R^{-1} \otimes I_n) x efficiently for 1D or 2D arrays."""
        is_2d = x.ndim == 2
        K = x.shape[1] if is_2d else 1

        if is_2d:  # multiple independent RHS columns not a 2D marginal residual vector
            x_mat = x.reshape((self.m, self.n * K))
            # R_inv_dense @ x_mat is much faster than cho_solve due to BLAS GEMM optimization
            # and returning a C-contiguous array which avoids an expensive copy during reshape.
            res = self.R_inv_dense @ x_mat
            return res.reshape(self.m * self.n, K)  # one reshape, not two
        else:
            x_mat = x.reshape((self.m, self.n))
            res = self.R_inv_dense @ x_mat
            return res.ravel()

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        is_2d = marginal_residual.ndim == 2
        K = marginal_residual.shape[1] if is_2d else 1

        # 1. Compute A^{-1} x = (R^{-1} \otimes I_n) x
        A_inv_x = self._apply_R_inv_kron(marginal_residual)

        if self.is_fast_path:
            v1 = self._build_v1(A_inv_x, is_2d)
            re = self.realized_effects[0]

            if is_2d:
                v1_b = (
                    v1.reshape(self.m, re.q, re.o, K)
                    .transpose(2, 0, 1, 3)
                    .reshape(re.o, self.m * re.q, K)
                )
            else:
                v1_b = (
                    v1.reshape(self.m, re.q, re.o)
                    .transpose(2, 0, 1)
                    .reshape(re.o, self.m * re.q, 1)
                )

            try:
                v2_b = np.linalg.solve(self.S_batch, v1_b)
            except np.linalg.LinAlgError:
                v2_b = np.empty_like(v1_b)
                for l in range(re.o):
                    v2_b[l] = pinvh(self.S_batch[l]) @ v1_b[l]

            v2_reshaped = v2_b.reshape(re.o, self.m, re.q, K).transpose(1, 2, 0, 3)
            v2 = (
                v2_reshaped.reshape(self.m * re.q * re.o, K)
                if is_2d
                else v2_reshaped.ravel()
            )

            v3 = self._build_v3(v2, is_2d, K)
            v4 = self._apply_R_inv_kron(v3)
            prec_resid = A_inv_x - v4

        elif self.S_factor is not None:
            # 2. Construct v1 = Z^T A^{-1} x
            v1 = self._build_v1(A_inv_x, is_2d)

            # 3. Solve S v2 = v1
            v2 = self.S_factor.solve(v1)
            if not is_2d and v2.ndim == 2 and v2.shape[1] == 1:
                v2 = v2.ravel()

            # 4. Compute v3 = Z v2
            v3 = self._build_v3(v2, is_2d, K)

            # 5. Compute v4 = A^{-1} v3
            v4 = self._apply_R_inv_kron(v3)

            prec_resid = A_inv_x - v4
        else:
            prec_resid = A_inv_x

        return prec_resid

    def logdet(self, **kwargs) -> float:
        """
        Compute exact log det(V) via the Matrix Determinant Lemma.

        Uses the identity (analogue of the Woodbury solve):

            log det(V) = log det(R⊗I_n) + log det(C) + log det(S)

        where:
          - log det(R⊗I_n) = n * log det(R)          — trivial, R is m×m
          - log det(C)      = Σ_k o_k * log det(D_k)  — trivial, D_k is (mq)×(mq)
          - log det(S)      = Σ log|diag(U_factor)|    — free, S already LU-factorized

        This is exact and deterministic; no SLQ probes needed.
        """
        n = self.n
        # --- Term 1: log det(R ⊗ I_n) = n * log det(R) ---
        R = self.realized_residual.term.cov
        sign_R, log_det_R = np.linalg.slogdet(R)
        if sign_R <= 0:
            warnings.warn(
                "Residual covariance is non-positive-definite; "
                "reverting to the best valid state.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.inf
        log_det_A = n * log_det_R

        # --- Term 2: log det(C) = Σ_k o_k * log det(D_k) ---
        log_det_C = 0.0
        for re in self.realized_effects:
            sign_Dk, log_det_Dk = np.linalg.slogdet(re.term.cov)
            if sign_Dk <= 0:
                warnings.warn(
                    f"Random effect covariance for group {re.term.group_id} is "
                    "non-positive-definite; reverting to the best valid state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return np.inf
            log_det_C += re.o * log_det_Dk

        # --- Term 3: log det(S) ---
        if self.is_fast_path:
            sign, logdets = np.linalg.slogdet(self.S_batch)
            if __debug__ and not np.all(sign > 0):
                warnings.warn(
                    "S matrix batch has non-positive determinants. "
                    "This indicates numerical instability or loss of positive-definiteness.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            log_det_S = np.sum(logdets)
        elif self.S_factor is not None:
            U_diag = self.S_factor.U.diagonal()
            if __debug__ and not np.all(U_diag > 0):
                warnings.warn(
                    "S matrix LU diagonal has non-positive entries. "
                    "This indicates numerical instability or loss of positive-definiteness.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            log_det_S = np.sum(np.log(np.abs(U_diag)))
        else:
            # No random effects: V = R ⊗ I_n, already fully captured by Term 1.
            log_det_S = 0.0

        return log_det_A + log_det_C + log_det_S


def build_solver(
    realized_effects: tuple[RealizedRandomEffect, ...],
    realized_residual: RealizedResidual,
    preconditioner: bool = True,
    cg_maxiter: int = 1000,
) -> BaseSolver:
    """Builds and returns the appropriate solver."""
    m = realized_residual.m
    n = realized_residual.n
    inner_dim = sum(re.o * re.q * m for re in realized_effects)  # generator, no list
    if inner_dim < m * n:
        return WoodburySolver(realized_effects, realized_residual)
    else:
        return IterativeSolver(
            realized_effects, realized_residual, preconditioner, cg_maxiter
        )
