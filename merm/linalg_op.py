import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import solve

# ====================== Linear Operator Implementations ======================

class VLinearOperator(LinearOperator):
    """
    A linear operator that represents the marginal covariance matrix V.
    V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)⁻ᵀ + R
    """
    def __init__(self, random_effects, resid_cov):
        self.random_effects = random_effects
        self.resid_cov = resid_cov
        self.n = list(random_effects.values())[0].n
        self.m = resid_cov.shape[0]
        shape = (self.m * self.n, self.m * self.n)
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x_vec):
        """Implements the matrix-vector product."""
        return V_matvec(x_vec, self.random_effects, self.resid_cov, self.n, self.m)
    
    def _adjoint(self):
        """Implements the adjoint operator V⁻ᵀ. Since V is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.random_effects, self.resid_cov))

# ===== PRECONDITIONER IMPLEMENTATIONS =========================================

class ResidualPreconditioner(LinearOperator):
    """
    The Lightweight Preconditioner: P = R = φ ⊗ Iₙ

    This is the simplest and cheapest preconditioner. It approximates the full
    covariance V with only its residual component R, ignoring all random effects.
    """
    def __init__(self, resid_cov, n):
        self.resid_cov = resid_cov
        self.n = n
        self.m = resid_cov.shape[0]
        shape = (self.m * self.n, self.m * self.n)
        self.resid_cov_inv = solve(self.resid_cov, np.eye(self.m), assume_a='pos')
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x_vec):
        """
        Computes (φ⁻¹ ⊗ Iₙ) @ x_vec.
        """
        x_mat = x_vec.reshape((self.n, self.m), order='F')
        result = x_mat @ self.resid_cov_inv
        return result.ravel(order='F')
    
    def _adjoint(self):
        """ Since P is symmetric, return self."""
        return self
    
    def __reduce__(self):
        """Enable pickling for multiprocessing."""
        return (self.__class__, (self.resid_cov, self.n))

# ====================== Matrix-Vector Operations ======================

def V_matvec(x_vec, random_effects, resid_cov, n, m):
    """
    Computes the marginal covariance matrix-vector product V @ x_vec,
        where V = Σ(Iₘ ⊗ Zₖ) Dₖ (Iₘ ⊗ Zₖ)⁻ᵀ + R.
    returns:
        1d array (Mn,)
    """
    Vx = cov_R_matvec(x_vec, resid_cov, n, m)
    for re in random_effects.values():
        np.add(Vx, cov_re_matvec(x_vec, re), out=Vx)
    return Vx.ravel(order='F')

def cov_R_matvec(x_vec, resid_cov, n, m):
    """
    Computes the residual covariance matrix-vector product (φ ⊗ Iₙ) @ x_vec.
    returns:
        2d array (n, m)
    """
    x_mat = x_vec.reshape((n, m), order='F')
    Rx = x_mat @ resid_cov
    return Rx

def cov_re_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product cov_re @ x_vec,
        where cov_re = (Iₘ ⊗ Z) D (Iₘ ⊗ Z)⁻ᵀ
    is random effect contribution to the marginal covariance.
    returns:
        2d array (n, m)
    """
    A_k = kronZ_T_matvec(x_vec, rand_effect)
    B_k = kronZ_D_matvec(A_k, rand_effect)
    return B_k

def kronZ_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (Iₘ ⊗ Z)⁻ᵀ @ x_vec maps a vector from
    the observation space back to the random effects space.
    returns:
        2d array (q*o, m)
    """
    x_mat = x_vec.reshape((rand_effect.n, rand_effect.m), order='F')
    A_k = rand_effect.Z.T @ x_mat
    return A_k

def kronZ_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product (Iₘ ⊗ Z) @ x_vec maps a vector from
    the random effects space to the observation space.
    returns:
        2d array (n, m)
    """
    x_mat = x_vec.reshape((rand_effect.q * rand_effect.o, rand_effect.m), order='F')
    A_k = rand_effect.Z @ x_mat
    return A_k

def cov_D_matvec(x_vec, rand_effect):
    """
    Computes the random effect covariance matrix-vector product (τ ⊗ Iₒ) @ x_vec.
    returns:
        2d array (o, m*q)
    """
    x_mat = x_vec.reshape((rand_effect.o, rand_effect.m * rand_effect.q), order='F')
    Dx =  x_mat @ rand_effect.cov
    return Dx

def kronZ_D_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W @ x_vec, where W = (Iₘ ⊗ Z) D maps a vector from
    the random effects space (pre-weighted by D) to the observation space.
    returns:
        2d array (n, m)
    """
    x_mat = x_vec.reshape((rand_effect.o, rand_effect.m * rand_effect.q), order='F')
    A_k = x_mat @ rand_effect.cov
    A_k = A_k.reshape((rand_effect.q * rand_effect.o, rand_effect.m), order='F')
    B_k = rand_effect.Z @ A_k
    return B_k

def kronZ_D_T_matvec(x_vec, rand_effect):
    """
    Computes the matrix-vector product W⁻ᵀ @ x_vec, where W⁻ᵀ = D (Iₘ ⊗ Z)⁻ᵀ maps a vector from
    the observation space to the random effects space (post-weighted by D).
    returns:
        2d array (o, m*q)
    """
    x_mat = x_vec.reshape((rand_effect.n, rand_effect.m), order='F')
    A_k = rand_effect.Z.T @ x_mat
    A_k = A_k.reshape((rand_effect.o, rand_effect.m * rand_effect.q), order='F')
    B_k = A_k @ rand_effect.cov
    return B_k

# ====================== Matrix-Matrix Operations ======================

def cov_R_matmat(x_mat, resid_cov, n, m):
    """
    Computes the residual covariance matrix-mat product (φ ⊗ Iₙ) @ x_mat.
    returns:
        3d array (n, n, num_cols)
    """
    num_cols = x_mat.shape[1]
    x_tensor = x_mat.reshape((n, m, num_cols), order='F')
    A_k_tensor = np.einsum('ijk,jl->ilk', x_tensor, resid_cov)
    return A_k_tensor

def kronZ_T_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product (Iₘ ⊗ Z)⁻ᵀ @ x_mat maps a matrix from
    the observation space to the random effects space.
    returns:
        3d array (q*o, n, num_cols)
    """
    num_cols = x_mat.shape[1]
    x_tensor = x_mat.reshape((rand_effect.n, rand_effect.m, num_cols), order='F')
    A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z.T, x_tensor)
    return A_k_tensor

def kronZ_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product (Iₘ ⊗ Z) @ x_mat maps a matrix from
    the random effects space to the observation space.
    returns:
        3d array (n, n, num_cols)
    """
    num_cols = x_mat.shape[1]
    # Reshape input into a 3D tensor (q*o, n, num_cols)
    x_tensor = x_mat.reshape((rand_effect.q * rand_effect.o, rand_effect.m, num_cols), order='F')
    # Apply the same Z matrix to each of the num_cols slices
    # 'ij,jkl->ikl' means: for each l, do matmul of (i,j) by (j,k)
    A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z, x_tensor)
    return A_k_tensor

def cov_D_matmat(x_mat, rand_effect):
    """
    Computes the random effect covariance matrix-matrix product (τ ⊗ Iₒ) @ x_mat.
    returns:
        3d array (o, m*q, num_cols)
    """
    num_cols = x_mat.shape[1]
    # Reshape input into a 3D tensor (o, m*q, num_cols)
    x_tensor = x_mat.reshape((rand_effect.o, rand_effect.m * rand_effect.q, num_cols), order='F')
    # Apply the same cov matrix to each of the num_cols slices
    # 'ijk,jl->ilk' means: for each k, do matrix multiply of (i,j) by (j,l)
    Dx_tensor = np.einsum('ijk,jl->ilk', x_tensor, rand_effect.cov)
    return Dx_tensor

def kronZ_D_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product W @ X, where W = (Iₘ ⊗ Z) D maps a matrix from
    the random effects space (pre-weighted by D) to the observation space.
    returns:
        3d array (n, n, num_cols)
    """
    num_cols = x_mat.shape[1]
    m, q, o = rand_effect.m, rand_effect.q, rand_effect.o
    # Step 1: Apply D (from cov_D_matmat)
    x_tensor = x_mat.reshape((o, m * q, num_cols), order='F')
    A_k_tensor = np.einsum('ijk,jl->ilk', x_tensor, rand_effect.cov) # Shape (o, m*q, num_cols)
    # Step 2: Apply (Iₘ ⊗ Z) (from kronZ_matmat)
    # Reshape the intermediate tensor for the final multiplication
    A_k_reshaped = A_k_tensor.reshape((q * o, m, num_cols), order='F')
    B_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z, A_k_reshaped)
    return B_k_tensor

def kronZ_D_T_matmat(x_mat, rand_effect):
    """
    Computes the matrix-matrix product W⁻ᵀ @ X, where W⁻ᵀ = D (Iₘ ⊗ Z)⁻ᵀ maps a matrix from
    the observation space to the random effects space (post-weighted by D).
    returns:
        3d array (o, m*q, num_cols)
    """
    num_cols = x_mat.shape[1]
    m, q, o, n = rand_effect.m, rand_effect.q, rand_effect.o, rand_effect.n
    x_tensor = x_mat.reshape((n, m , num_cols), order='F')
    A_k_tensor = np.einsum('ij,jkl->ikl', rand_effect.Z.T, x_tensor)
    A_k_reshaped = A_k_tensor.reshape((o, m*q, num_cols), order='F')
    B_k_tensor = np.einsum('ijk,jl->ilk', A_k_reshaped, rand_effect.cov)
    return B_k_tensor