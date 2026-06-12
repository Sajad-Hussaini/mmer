import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import cho_solve
from ..structures.covariance import RandomCovariance, ResidualCovariance
from ..structures.design import RandomDesign


def generate_rademacher_probes(
    n_rows: int, n_probes: int, seed: int = 42
) -> np.ndarray:
    """Generate a Rademacher ±1 probe matrix directly in float64.

    The original implementation used ``rng.integers(..., dtype=np.intp)`` which
    creates an int64 intermediate, then ``.astype(np.float64, copy=False)`` which
    always copies because int64 ≠ float64 (``copy=False`` only skips copies when
    dtypes already match).

    This version uses ``rng.integers(dtype=np.int8)`` to produce an 8× smaller
    intermediate (int8 vs int64), then converts to float64 and scales in-place.
    ``rng.integers`` does not accept ``dtype=float64`` directly.
    """
    rng = np.random.default_rng(seed)
    out = rng.integers(0, 2, size=(n_rows, n_probes), dtype=np.int8).astype(np.float64)
    out *= 2.0
    out -= 1.0
    return out


class KroneckerMath:
    """
    Pure mathematical operators for Kronecker matrix-vector products.
    Decoupled from data structures and states.
    """

    @staticmethod
    def _kronZ_T_matvec(
        Z,
        x_vec: np.ndarray,
        n_responses: int,
        n_effects: int,
        n_levels: int,
        n_samples: int,
        out=None,
    ):
        """(I_m ⊗ Z^T) @ x"""
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((n_responses, n_samples, K))
            if out is None:
                out = np.empty((n_responses, n_effects * n_levels, K))
            out_r = out.reshape((n_responses, n_effects * n_levels, K))
            for i in range(n_responses):
                out_r[i] = Z.T @ xr[i]
            return out_r.reshape(n_responses * n_effects * n_levels, K)
        else:
            xr = x_vec.reshape((n_responses, n_samples))
            if out is None:
                out = np.empty((n_responses, n_effects * n_levels))
            out_r = out.reshape((n_responses, n_effects * n_levels))
            for i in range(n_responses):
                out_r[i] = Z.T @ xr[i]
            return out.ravel()

    @staticmethod
    def _kronZ_matvec(
        Z,
        x_vec: np.ndarray,
        n_responses: int,
        n_effects: int,
        n_levels: int,
        n_samples: int,
        out=None,
    ):
        """(I_m ⊗ Z) @ x"""
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((n_responses, n_effects * n_levels, K))
            if out is None:
                out = np.empty((n_responses, n_samples, K))
            out_r = out.reshape((n_responses, n_samples, K))
            for i in range(n_responses):
                out_r[i] = Z @ xr[i]
            return out_r.reshape(n_responses * n_samples, K)
        else:
            xr = x_vec.reshape((n_responses, n_effects * n_levels))
            if out is None:
                out = np.empty((n_responses, n_samples))
            out_r = out.reshape((n_responses, n_samples))
            for i in range(n_responses):
                out_r[i] = Z @ xr[i]
            return out.ravel()

    @staticmethod
    def _D_matvec(
        cov_mat: np.ndarray,
        x_vec: np.ndarray,
        n_responses: int,
        n_effects: int,
        n_levels: int,
        out: np.ndarray | None = None,
    ):
        """D @ x"""
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((n_responses * n_effects, n_levels, K))
            xr_flat = xr.reshape((n_responses * n_effects, n_levels * K))

            if out is None:
                out = np.empty((n_responses * n_effects * n_levels, K))
            out_r = out.reshape((n_responses * n_effects, n_levels * K))

            np.dot(cov_mat, xr_flat, out=out_r)
            return out.reshape(n_responses * n_effects * n_levels, K)
        else:
            xr = x_vec.reshape((n_responses * n_effects, n_levels))
            if out is not None:
                out_r = out.reshape((n_responses * n_effects, n_levels))
                np.dot(cov_mat, xr, out=out_r)
                return out
            return np.dot(cov_mat, xr).ravel()

    @classmethod
    def _kronZ_D_T_matvec(
        cls,
        Z,
        cov_mat,
        x_vec,
        n_responses,
        n_effects,
        n_levels,
        n_samples,
        out_A=None,
        out_D=None,
    ):
        """D (I_m ⊗ Z^T) @ x"""
        A_k = cls._kronZ_T_matvec(
            Z, x_vec, n_responses, n_effects, n_levels, n_samples, out=out_A
        )
        return cls._D_matvec(cov_mat, A_k, n_responses, n_effects, n_levels, out=out_D)

    @classmethod
    def full_cov_matvec(
        cls,
        Z,
        cov_mat,
        x_vec,
        n_responses,
        n_effects,
        n_levels,
        n_samples,
        out=None,
        buf_A=None,
        buf_D=None,
        buf_ans=None,
    ):
        """(I_m ⊗ Z) D (I_m ⊗ Z^T) @ x"""
        A_k = cls._kronZ_D_T_matvec(
            Z,
            cov_mat,
            x_vec,
            n_responses,
            n_effects,
            n_levels,
            n_samples,
            out_A=buf_A,
            out_D=buf_D,
        )

        ans = cls._kronZ_matvec(
            Z, A_k, n_responses, n_effects, n_levels, n_samples, out=buf_ans
        )
        if out is not None:
            out += ans
            return out
        return ans

    @staticmethod
    def resid_full_cov_matvec(
        cov_mat: np.ndarray,
        x_vec: np.ndarray,
        n_responses: int,
        n_samples: int,
        out: np.ndarray | None = None,
    ):
        """(R ⊗ I_n) @ x"""
        is_2d = x_vec.ndim == 2
        K = x_vec.shape[1] if is_2d else 1

        if is_2d:
            xr = x_vec.reshape((n_responses, n_samples * K))
            if out is not None:
                out_r = out.reshape((n_responses, n_samples * K))
                np.dot(cov_mat, xr, out=out_r)
                return out.reshape((n_responses * n_samples, K))
            ans = np.dot(cov_mat, xr).reshape(n_responses * n_samples, K)
        else:
            xr = x_vec.reshape((n_responses, n_samples))
            if out is not None:
                out_r = out.reshape((n_responses, n_samples))
                np.dot(cov_mat, xr, out=out_r)
                return out
            ans = np.dot(cov_mat, xr).ravel()
        return ans


class VLinearOperator(LinearOperator):
    """
    Linear Operator for the marginal covariance matrix V.
    V = Sum (I_m \\otimes Z_k) D_k (I_m \\otimes Z_k)^T + R
    """

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
        super().__init__(
            dtype=np.float64,
            shape=(
                self.n_responses * self.n_samples,
                self.n_responses * self.n_samples,
            ),
        )

        # Pre-allocate workspaces
        self._buf_A_mqo_list = [
            np.empty(cov.n_responses * cov.n_effects * d.n_levels)
            for cov, d in zip(self.random_covs, self.random_designs)
        ]
        self._buf_D_mqo_list = [
            np.empty(cov.n_responses * cov.n_effects * d.n_levels)
            for cov, d in zip(self.random_covs, self.random_designs)
        ]
        self._buf_Vx = np.empty(self.n_responses * self.n_samples)
        self._buf_ans = np.empty(self.n_responses * self.n_samples)
        self._buf_mat_A_mqo_list = None
        self._buf_mat_D_mqo_list = None
        self._buf_mat_ans = None

    def _matvec(self, x_vec: np.ndarray):
        Vx = KroneckerMath.resid_full_cov_matvec(
            self.resid_cov.matrix, x_vec, self.n_responses, self.n_samples, out=None
        )
        for i, (cov, d) in enumerate(zip(self.random_covs, self.random_designs)):
            KroneckerMath.full_cov_matvec(
                d.Z,
                cov.matrix,
                x_vec,
                cov.n_responses,
                cov.n_effects,
                d.n_levels,
                self.n_samples,
                out=Vx,
                buf_A=self._buf_A_mqo_list[i],
                buf_D=self._buf_D_mqo_list[i],
                buf_ans=self._buf_ans,
            )
        return Vx

    def _matmat(self, x_mat: np.ndarray):
        K = x_mat.shape[1]
        if self._buf_mat_A_mqo_list is None or (
            # Guard against IndexError when random_covs is empty (no random
            # effects — pure fixed-effects model).
            len(self._buf_mat_A_mqo_list) > 0
            and self._buf_mat_A_mqo_list[0].shape[1] != K
        ):
            self._buf_mat_A_mqo_list = [
                np.empty((cov.n_responses * cov.n_effects * d.n_levels, K))
                for cov, d in zip(self.random_covs, self.random_designs)
            ]
            self._buf_mat_D_mqo_list = [
                np.empty((cov.n_responses * cov.n_effects * d.n_levels, K))
                for cov, d in zip(self.random_covs, self.random_designs)
            ]
            self._buf_mat_ans = np.empty((self.n_responses * self.n_samples, K))

        Vx = KroneckerMath.resid_full_cov_matvec(
            self.resid_cov.matrix, x_mat, self.n_responses, self.n_samples, out=None
        )
        for i, (cov, d) in enumerate(zip(self.random_covs, self.random_designs)):
            KroneckerMath.full_cov_matvec(
                d.Z,
                cov.matrix,
                x_mat,
                cov.n_responses,
                cov.n_effects,
                d.n_levels,
                self.n_samples,
                out=Vx,
                buf_A=self._buf_mat_A_mqo_list[i],
                buf_D=self._buf_mat_D_mqo_list[i],
                buf_ans=self._buf_mat_ans,
            )
        return Vx

    def _adjoint(self):
        return self


class ResidualPreconditioner(LinearOperator):
    def __init__(
        self,
        resid_cov_inv: np.ndarray | None,
        resid_cov_chol: tuple | None,
        n_samples: int,
        n_responses: int,
    ):
        self.cov_inv = resid_cov_inv
        self.cov_chol = resid_cov_chol
        self.n_samples = n_samples
        self.n_responses = n_responses
        super().__init__(
            dtype=np.float64,
            shape=(
                self.n_responses * self.n_samples,
                self.n_responses * self.n_samples,
            ),
        )

    def _matvec(self, x_vec: np.ndarray):
        x_mat = x_vec.reshape((self.n_responses, self.n_samples))
        if self.cov_chol is not None:
            Px = cho_solve(self.cov_chol, x_mat, check_finite=False).ravel()
        else:
            Px = (self.cov_inv @ x_mat).ravel()
        return Px

    def _matmat(self, x_mat: np.ndarray):
        K = x_mat.shape[1]
        x_reshaped = x_mat.reshape((self.n_responses, self.n_samples * K))
        if self.cov_chol is not None:
            Px = cho_solve(self.cov_chol, x_reshaped, check_finite=False)
        else:
            Px = self.cov_inv @ x_reshaped
        Px = Px.reshape(self.n_responses * self.n_samples, K)
        return Px

    def _adjoint(self):
        return self
