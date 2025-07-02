import numpy as np
from scipy.sparse.linalg import LinearOperator

class VLinearOperator(LinearOperator):
    def __init__(self, random_effects, resid_cov, M, n):
        self.random_effects = random_effects
        self.resid_cov = resid_cov
        self.M = M
        self.n = n
        shape = (M * n, M * n)
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x_vec):
        """Implements the matrix-vector product."""
        return V_matvec(x_vec, self.random_effects, self.resid_cov, self.M, self.n)
    
    def _adjoint(self):
        """Implements the adjoint operator V^T. Since V is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.random_effects, self.resid_cov, self.M, self.n))

def V_matvec(x_vec, random_effects, resid_cov, M, n):
    """
    Computes the margianl covariance matrix-vector product V @ x_vec,
        where V = Σ(I_M ⊗ Z_k) D_k (I_M ⊗ Z_k)^T + R.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    Vx = cov_R_matvec(x_vec, resid_cov, M, n)
    for re in random_effects.values():
        np.add(Vx, cov_re_matvec(x_vec, re), out=Vx)
    return Vx  # (M*n, )

def cov_re_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product cov_re @ x_vec,
        where cov_re = (I_M ⊗ Z) D (I_M ⊗ Z)^T is random effect contribution to the marginal covariance.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    A_k = kronZ_T_matvec(x_vec, rand_effect)
    Dx = cov_D_matvec(A_k, rand_effect)
    B_k = kronZ_matvec(Dx, rand_effect)
    return B_k  # (M*n,)

def kronZ_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (I_M ⊗ Z) @ x_vec maps a vector from
    the random effects space to the observation space.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_effect * rand_effect.n_level, rand_effect.n_res), order='F')
    A_k = rand_effect.Z @ x_mat
    return A_k.ravel(order='F')  # (M*n, )

def kronZ_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (I_M ⊗ Z)^T @ x_vec maps a vector from
    the observation space back to the random effects space.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_obs, rand_effect.n_res), order='F')
    A_k = rand_effect.Z.T @ x_mat
    return A_k.ravel(order='F')  # (M*q*o, )

def cov_D_matvec(x_vec, rand_effect):
    """
    Computes the random effect covariance matrix-vector product (Tau ⊗ I_o) @ x_vec.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_level, rand_effect.n_res * rand_effect.n_effect), order='F')
    Dx =  x_mat @ rand_effect.cov
    return Dx.ravel(order='F')  # (M*q*o, )

def cov_R_matvec(x_vec, resid_cov, M, n):
    """
    Computes the residual covaraince matrix-vector product (Phi ⊗ I_n) @ x_vec.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((n, M), order='F')
    Rx = x_mat @ resid_cov
    return Rx.ravel(order='F')  # (M*n, )

def W_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W @ x_vec, where W = (I_M ⊗ Z) D maps a vector from
    the random effects space (pre-weighted by D) to the observation space.
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_res * rand_effect.n_effect, rand_effect.n_level)).T
    A_k = x_mat @ rand_effect.cov
    A_k = A_k.reshape((rand_effect.n_level, rand_effect.n_res, rand_effect.n_effect)).transpose(1, 2, 0).reshape((rand_effect.n_res, rand_effect.n_effect * rand_effect.n_level)).T
    B_k = rand_effect.Z @ A_k
    return B_k.T.ravel()  # (M*n, )

def W_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W^T @ x_vec, where W^T = D (I_M ⊗ Z)^T maps a vector from
    the observation space back to the random effects space (post-weighted by D).
    It leverages the Kronecker structure to avoid full matrix construction.
    """
    x_mat = x_vec.reshape((rand_effect.n_res, rand_effect.n_obs)).T
    A_k = rand_effect.Z.T @ x_mat
    A_k = A_k.reshape((rand_effect.n_effect, rand_effect.n_level, rand_effect.n_res)).transpose(1, 2, 0).reshape((rand_effect.n_level, rand_effect.n_res * rand_effect.n_effect))
    B_k = A_k @ rand_effect.cov
    B_k = B_k.reshape((rand_effect.n_level, rand_effect.n_res, rand_effect.n_effect)).transpose(1, 2, 0).ravel()  # (M*q*o, )
    return B_k