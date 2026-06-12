import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu
from scipy.linalg import cho_factor, cho_solve, LinAlgError, pinvh
from . import lanczos
from .operator import VLinearOperator, ResidualPreconditioner, KroneckerMath
from ..structures.covariance import RandomCovariance, ResidualCovariance
from ..structures.design import RandomDesign


def _get_chol_or_inv(mat: np.ndarray) -> tuple:
    try:
        return cho_factor(mat, lower=True, check_finite=False), None
    except LinAlgError:
        return None, pinvh(mat)


def _invert_matrix(mat: np.ndarray) -> np.ndarray:
    chol, inv = _get_chol_or_inv(mat)
    if chol is not None:
        return cho_solve(chol, np.eye(mat.shape[0]), check_finite=False)
    return inv


class BaseSolver:
    is_iterative: bool = False

    def __init__(
        self,
        random_covs: tuple[RandomCovariance, ...],
        random_designs: tuple[RandomDesign, ...],
        resid_cov: ResidualCovariance,
        n: int,
    ):
        self.random_covs = random_covs
        self.random_designs = random_designs
        self.resid_cov = resid_cov
        self.n_samples = n
        self.n_responses = resid_cov.n_responses
        self.V_op = VLinearOperator(
            self.random_covs, self.random_designs, self.resid_cov, self.n_samples
        )

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def logdet(self, **kwargs) -> float:
        raise NotImplementedError


class IterativeSolver(BaseSolver):
    is_iterative: bool = True

    def __init__(
        self,
        random_covs: tuple[RandomCovariance, ...],
        random_designs: tuple[RandomDesign, ...],
        resid_cov: ResidualCovariance,
        n: int,
        preconditioner: bool = True,
        cg_maxiter: int = 1000,
    ):
        super().__init__(random_covs, random_designs, resid_cov, n)
        self.use_preconditioner = preconditioner
        self.cg_maxiter = cg_maxiter

        self.M_op = None
        if self.use_preconditioner:
            R = self.resid_cov.matrix
            R_chol, R_inv = _get_chol_or_inv(R)
            self.M_op = ResidualPreconditioner(
                R_inv if R_chol is None else None,
                R_chol,
                self.n_samples,
                self.n_responses,
            )

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        if marginal_residual.ndim == 2:
            preconditioned_residuals = np.empty_like(marginal_residual)
            for i, rhs in enumerate(marginal_residual.T):
                sol, info = cg(A=self.V_op, b=rhs, M=self.M_op, maxiter=self.cg_maxiter)
                if info < 0:
                    raise RuntimeError(f"Conjugate Gradient breakdown (info={info}).")
                elif info > 0:
                    import warnings

                    warnings.warn(f"Conjugate Gradient did not converge (info={info}).")
                preconditioned_residuals[:, i] = sol
        else:
            preconditioned_residuals, info = cg(
                A=self.V_op, b=marginal_residual, M=self.M_op, maxiter=self.cg_maxiter
            )
            if info < 0:
                raise RuntimeError(f"Conjugate Gradient breakdown (info={info}).")
            elif info > 0:
                import warnings

                warnings.warn(f"Conjugate Gradient did not converge (info={info}).")

        return preconditioned_residuals

    def logdet(
        self,
        slq_steps: int,
        n_probes: int,
        n_jobs: int = -1,
        backend: str = "threading",
    ) -> float:
        return lanczos.logdet(self.V_op, slq_steps, n_probes, n_jobs, backend)


class WoodburySolver(BaseSolver):
    def __init__(
        self,
        random_covs: tuple[RandomCovariance, ...],
        random_designs: tuple[RandomDesign, ...],
        resid_cov: ResidualCovariance,
        n: int,
    ):
        super().__init__(random_covs, random_designs, resid_cov, n)

        R = self.resid_cov.matrix
        self.R_inv_dense = _invert_matrix(R)
        R_inv_sp = sparse.csr_array(self.R_inv_dense)

        self.is_fast_path = len(self.random_covs) == 1

        if self.is_fast_path:
            cov = self.random_covs[0]
            d = self.random_designs[0]

            ZTZ_coo = d.ZTZ.tocoo()
            u_idx = ZTZ_coo.row // d.n_levels
            l_idx = ZTZ_coo.row % d.n_levels
            v_idx = ZTZ_coo.col // d.n_levels

            A = np.zeros((d.n_levels, cov.n_effects, cov.n_effects))
            A[l_idx, u_idx, v_idx] = ZTZ_coo.data

            S_kron = np.einsum("ij,luv->liujv", self.R_inv_dense, A).reshape(
                d.n_levels,
                self.n_responses * cov.n_effects,
                self.n_responses * cov.n_effects,
            )
            D_inv_dense = _invert_matrix(cov.matrix)
            self.S_batch = S_kron + D_inv_dense
            self.S_factor = None
        else:
            self.S_batch = None
            self.S_chol = None
            self.S_dense_fallback = None

            k_len = len(self.random_covs)
            S_blocks = [[None] * k_len for _ in range(k_len)]
            for i, (cov_i, d_i) in enumerate(
                zip(self.random_covs, self.random_designs)
            ):
                for j, (cov_j, d_j) in enumerate(
                    zip(self.random_covs, self.random_designs)
                ):
                    if j < i:
                        S_ij = S_blocks[j][i].T
                    else:
                        Z_i_T_Z_j = d_i.get_Z_cross_product(d_j)
                        S_ij = sparse.kron(R_inv_sp, Z_i_T_Z_j)

                        if i == j:
                            D_inv = _invert_matrix(cov_i.matrix)
                            I_oi = sparse.eye_array(d_i.n_levels, format="csr")
                            C_inv_ii = sparse.kron(sparse.csr_array(D_inv), I_oi)
                            S_ij = S_ij + C_inv_ii

                    S_blocks[i][j] = S_ij

            if S_blocks:
                S_mat = sparse.block_array(S_blocks, format="csc")
                self.S_factor = splu(S_mat)
            else:
                self.S_factor = None

    def _build_v1(self, A_inv_x: np.ndarray, is_2d: bool) -> np.ndarray:
        v1_list = [
            KroneckerMath._kronZ_T_matvec(
                d.Z, A_inv_x, cov.n_responses, cov.n_effects, d.n_levels, self.n_samples
            )
            for cov, d in zip(self.random_covs, self.random_designs)
        ]
        if len(v1_list) == 1:
            return v1_list[0]
        return np.vstack(v1_list) if is_2d else np.concatenate(v1_list)

    def _build_v3(self, v2: np.ndarray, is_2d: bool, K: int) -> np.ndarray:
        if len(self.random_covs) == 1:
            cov, d = self.random_covs[0], self.random_designs[0]
            return KroneckerMath._kronZ_matvec(
                d.Z, v2, cov.n_responses, cov.n_effects, d.n_levels, self.n_samples
            )

        if is_2d:
            v3 = np.zeros((self.n_responses * self.n_samples, K))
        else:
            v3 = np.zeros(self.n_responses * self.n_samples)

        offset = 0
        for cov, d in zip(self.random_covs, self.random_designs):
            size_i = self.n_responses * cov.n_effects * d.n_levels
            v2_i = v2[offset : offset + size_i]
            v3 += KroneckerMath._kronZ_matvec(
                d.Z, v2_i, cov.n_responses, cov.n_effects, d.n_levels, self.n_samples
            )
            offset += size_i
        return v3

    def _apply_R_inv_kron(self, x: np.ndarray) -> np.ndarray:
        is_2d = x.ndim == 2
        K = x.shape[1] if is_2d else 1
        if is_2d:
            x_mat = x.reshape((self.n_responses, self.n_samples * K))
            res = self.R_inv_dense @ x_mat
            return res.reshape(self.n_responses * self.n_samples, K)
        else:
            x_mat = x.reshape((self.n_responses, self.n_samples))
            res = self.R_inv_dense @ x_mat
            return res.ravel()

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        is_2d = marginal_residual.ndim == 2
        K = marginal_residual.shape[1] if is_2d else 1

        A_inv_x = self._apply_R_inv_kron(marginal_residual)

        if self.is_fast_path:
            v1 = self._build_v1(A_inv_x, is_2d)
            cov = self.random_covs[0]
            d = self.random_designs[0]

            if is_2d:
                v1_b = (
                    v1.reshape(self.n_responses, cov.n_effects, d.n_levels, K)
                    .transpose(2, 0, 1, 3)
                    .reshape(d.n_levels, self.n_responses * cov.n_effects, K)
                )
            else:
                v1_b = (
                    v1.reshape(self.n_responses, cov.n_effects, d.n_levels)
                    .transpose(2, 0, 1)
                    .reshape(d.n_levels, self.n_responses * cov.n_effects, 1)
                )

            try:
                v2_b = np.linalg.solve(self.S_batch, v1_b)
            except np.linalg.LinAlgError:
                v2_b = np.empty_like(v1_b)
                for idx in range(d.n_levels):
                    v2_b[idx] = pinvh(self.S_batch[idx]) @ v1_b[idx]

            v2_reshaped = v2_b.reshape(
                d.n_levels, self.n_responses, cov.n_effects, K
            ).transpose(1, 2, 0, 3)
            v2 = (
                v2_reshaped.reshape(self.n_responses * cov.n_effects * d.n_levels, K)
                if is_2d
                else v2_reshaped.ravel()
            )

            v3 = self._build_v3(v2, is_2d, K)
            v4 = self._apply_R_inv_kron(v3)
            preconditioned_residuals = A_inv_x - v4

        elif self.S_factor is not None:
            v1 = self._build_v1(A_inv_x, is_2d)
            v2 = self.S_factor.solve(v1)
            if not is_2d and v2.ndim == 2 and v2.shape[1] == 1:
                v2 = v2.ravel()

            v3 = self._build_v3(v2, is_2d, K)
            v4 = self._apply_R_inv_kron(v3)
            preconditioned_residuals = A_inv_x - v4
        else:
            preconditioned_residuals = A_inv_x

        return preconditioned_residuals

    def logdet(self, **kwargs) -> float:
        n = self.n_samples
        R = self.resid_cov.matrix
        sign_R, log_det_R = np.linalg.slogdet(R)
        if sign_R <= 0:
            return np.inf
        log_det_A = n * log_det_R

        log_det_C = 0.0
        for cov, d in zip(self.random_covs, self.random_designs):
            sign_Dk, log_det_Dk = np.linalg.slogdet(cov.matrix)
            if sign_Dk <= 0:
                return np.inf
            log_det_C += d.n_levels * log_det_Dk

        if self.is_fast_path:
            sign, logdets = np.linalg.slogdet(self.S_batch)
            if np.any(sign <= 0):
                return np.inf
            log_det_S = np.sum(logdets)
        elif self.S_factor is not None:
            U_diag = self.S_factor.U.diagonal()
            log_det_S = np.sum(np.log(np.abs(U_diag)))
        else:
            log_det_S = 0.0

        return log_det_A + log_det_C + log_det_S


def build_solver(
    random_covs: tuple[RandomCovariance, ...],
    random_designs: tuple[RandomDesign, ...],
    resid_cov: ResidualCovariance,
    n: int,
    preconditioner: bool = True,
    cg_maxiter: int = 1000,
    force_iterative: bool = False,
) -> BaseSolver:

    if force_iterative:
        return IterativeSolver(
            random_covs, random_designs, resid_cov, n, preconditioner, cg_maxiter
        )

    n_responses = resid_cov.n_responses
    inner_dim = sum(
        d.n_levels * cov.n_effects * n_responses
        for cov, d in zip(random_covs, random_designs)
    )

    if inner_dim < n_responses * n:
        try:
            return WoodburySolver(random_covs, random_designs, resid_cov, n)
        except (MemoryError, RuntimeError):
            pass

    return IterativeSolver(
        random_covs, random_designs, resid_cov, n, preconditioner, cg_maxiter
    )
