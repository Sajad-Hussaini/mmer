import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve

class VLinearOperator(LinearOperator):
    """
    A linear operator that represents the marginal covariance matrix V.
    V = Σ(I_M ⊗ Z_k) D_k (I_M ⊗ Z_k)^T + R
    """
    def __init__(self, random_effects, resid_cov):
        self.random_effects = random_effects
        self.resid_cov = resid_cov
        self.M = resid_cov.shape[0]
        self.n = list(random_effects.values())[0].n_obs
        shape = (self.M * self.n, self.M * self.n)
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x_vec):
        """Implements the matrix-vector product."""
        return V_matvec(x_vec, self.random_effects, self.resid_cov, self.M, self.n)
    
    def _adjoint(self):
        """Implements the adjoint operator V^T. Since V is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.random_effects, self.resid_cov))

class ResidualPreconditioner(LinearOperator):
    """
    A preconditioner for V using the inverse of the residual covariance R.
    P⁻¹ = R⁻¹ = φ⁻¹ ⊗ Iₙ
    """
    def __init__(self, resid_cov, n):
        self.resid_cov = resid_cov
        self.n = n
        self.M = resid_cov.shape[0]
        shape = (self.M * self.n, self.M * self.n)
        self.resid_cov_inv = solve(self.resid_cov, np.eye(self.M), assume_a='pos')
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x_vec):
        """
        Applies the preconditioner P⁻¹ to a vector v.
        This computes (φ⁻¹ ⊗ Iₙ) * v.
        """
        x_mat = x_vec.reshape((self.n, self.M), order='F')
        result = x_mat @ self.resid_cov_inv
        return result.ravel(order='F')
    
    def _adjoint(self):
        """Implements the adjoint operator P^T. Since P is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.resid_cov, self.n))

# ====================== Matrix-Vector Operations ======================

def V_matvec(x_vec, random_effects, resid_cov, M, n):
    """
    Computes the margianl covariance matrix-vector product V @ x_vec,
        where V = Σ(I_M ⊗ Z_k) D_k (I_M ⊗ Z_k)^T + R.
    returns:
        1d array (Mn,)
    """
    Vx = cov_R_matvec(x_vec, resid_cov, M, n)
    for re in random_effects.values():
        np.add(Vx, cov_re_matvec(x_vec, re), out=Vx)
    return Vx.ravel(order='F')

def cov_R_matvec(x_vec, resid_cov, M, n):
    """
    Computes the residual covariance matrix-vector product (Phi ⊗ I_n) @ x_vec.
    returns:
        2d array (n, M)
    """
    x_mat = x_vec.reshape((n, M), order='F')
    Rx = x_mat @ resid_cov
    return Rx

def cov_re_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product cov_re @ x_vec,
        where cov_re = (I_M ⊗ Z) D (I_M ⊗ Z)^T is random effect contribution to the marginal covariance.
    returns:
        2d array (n, M)
    """
    A_k = kronZ_T_matvec(x_vec, rand_effect)
    B_k = kronZ_D_matvec(A_k, rand_effect)
    return B_k

def kronZ_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (I_M ⊗ Z)^T @ x_vec maps a vector from
    the observation space back to the random effects space.
    returns:
        2d arrray (q*o, M)
    """
    x_mat = x_vec.reshape((rand_effect.n_obs, rand_effect.n_res), order='F')
    A_k = rand_effect.Z.T @ x_mat
    return A_k

def kronZ_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (I_M ⊗ Z) @ x_vec maps a vector from
    the random effects space to the observation space.
    returns:
        2d array (n, M)
    """
    x_mat = x_vec.reshape((rand_effect.n_effect * rand_effect.n_level, rand_effect.n_res), order='F')
    A_k = rand_effect.Z @ x_mat
    return A_k

def cov_D_matvec(x_vec, rand_effect):
    """
    Computes the random effect covariance matrix-vector product (Tau ⊗ I_o) @ x_vec.
    returns:
        2d array (o, M*q)
    """
    x_mat = x_vec.reshape((rand_effect.n_level, rand_effect.n_res * rand_effect.n_effect), order='F')
    Dx =  x_mat @ rand_effect.cov
    return Dx

def kronZ_D_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W @ x_vec, where W = (I_M ⊗ Z) D maps a vector from
    the random effects space (pre-weighted by D) to the observation space.
    returns:
        2d array (n, M)
    """
    x_mat = x_vec.reshape((rand_effect.n_level, rand_effect.n_res * rand_effect.n_effect), order='F')
    A_k = x_mat @ rand_effect.cov
    A_k = A_k.reshape((rand_effect.n_effect * rand_effect.n_level, rand_effect.n_res), order='F')
    B_k = rand_effect.Z @ A_k
    return B_k

def kronZ_D_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W^T @ x_vec, where W^T = D (I_M ⊗ Z)^T maps a vector from
    the observation space to the random effects space (post-weighted by D).
    returns:
        2d array (o, M*q)
    """
    x_mat = x_vec.reshape((rand_effect.n_obs, rand_effect.n_res), order='F')
    A_k = rand_effect.Z.T @ x_mat
    A_k = A_k.reshape((rand_effect.n_level, rand_effect.n_res * rand_effect.n_effect), order='F')
    B_k = A_k @ rand_effect.cov
    return B_k

# ====================== Matrix-Matrix Operations ======================

def cov_R_matmat(x_mat, resid_cov, M, n):
    """
    Computes the residual covariance matrix-mat product (Phi ⊗ I_n) @ x_mat.
    returns:
        3d array (n, M, num_cols)
    """
    num_cols = x_mat.shape[1]
    x_tensor = x_mat.reshape((n, M, num_cols), order='F')
    A_k_tensor = np.einsum('ijk,jl->ilk', x_tensor, resid_cov)
    return A_k_tensor

def kronZ_T_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product (I_M ⊗ Z)^T @ x_mat maps a matrix from
    the observation space to the random effects space.
    returns:
        3d array (q*o, M, num_cols)
    """
    num_cols = x_mat.shape[1]
    M, n = rand_effect.n_res, rand_effect.n_obs
    x_tensor = x_mat.reshape((n, M, num_cols), order='F')
    A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z.T, x_tensor)
    return A_k_tensor

def kronZ_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product (I_M ⊗ Z) @ x_mat maps a matrix from
    the random effects space to the observation space.
    returns:
        3d array (n, M, num_cols)
    """
    num_cols = x_mat.shape[1]
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    # Reshape input into a 3D tensor (q*o, M, num_cols)
    x_tensor = x_mat.reshape((q * o, M, num_cols), order='F')
    # Apply the same Z matrix to each of the num_cols slices
    # 'ij,jkl->ikl' means: for each l, do matmul of (i,j) by (j,k)
    A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z, x_tensor)
    return A_k_tensor

def cov_D_matmat(x_mat, rand_effect):
    """
    Computes the random effect covariance matrix-matrix product (Tau ⊗ I_o) @ x_mat.
    returns:
        3d array (o, M*q, num_cols)
    """
    num_cols = x_mat.shape[1]
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    # Reshape input into a 3D tensor (o, M*q, num_cols)
    x_tensor = x_mat.reshape((o, M * q, num_cols), order='F')
    # Apply the same cov matrix to each of the num_cols slices
    # 'ijk,jl->ilk' means: for each k, do matrix multiply of (i,j) by (j,l)
    Dx_tensor = np.einsum('ijk,jl->ilk', x_tensor, rand_effect.cov)
    return Dx_tensor

def kronZ_D_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product W @ X, where W = (I_M ⊗ Z) D maps a matrix from
    the random effects space (pre-weighted by D) to the observation space.
    returns:
        3d array (n, M, num_cols)
    """
    num_cols = x_mat.shape[1]
    M, q, o = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level
    # Step 1: Apply D (from cov_D_matmat)
    x_tensor = x_mat.reshape((o, M * q, num_cols), order='F')
    A_k_tensor = np.einsum('ijk,jl->ilk', x_tensor, rand_effect.cov) # Shape (o, M*q, num_cols)
    # Step 2: Apply (I_M ⊗ Z) (from kronZ_matmat)
    # Reshape the intermediate tensor for the final multiplication
    A_k_reshaped = A_k_tensor.reshape((q * o, M, num_cols), order='F')
    B_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z, A_k_reshaped)
    return B_k_tensor

def kronZ_D_T_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product W^T @ X, where W^T = D (I_M ⊗ Z)^T maps a matrix from
    the observation space to the random effects space (post-weighted by D).
    returns:
        3d array (o, M*q, num_cols)
    """
    num_cols = x_mat.shape[1]
    M, q, o, n = rand_effect.n_res, rand_effect.n_effect, rand_effect.n_level, rand_effect.n_obs
    x_tensor = x_mat.reshape((n, M , num_cols), order='F')
    A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z.T, x_tensor)
    A_k_reshaped = A_k_tensor.reshape((o, M*q, num_cols), order='F')
    B_k_tensor = np.einsum('ijk,jl->ilk', A_k_reshaped, rand_effect.cov)
    return B_k_tensor