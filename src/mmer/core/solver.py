import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve
from scipy.linalg import solve
from .operator import VLinearOperator, ResidualPreconditioner
from .terms import RealizedRandomEffect, RealizedResidual

def _invert_matrix(mat: np.ndarray) -> np.ndarray:
    """Safely invert a symmetric matrix, falling back to standard inverse if Cholesky fails."""
    try:
        return solve(a=mat, b=np.eye(mat.shape[0]), assume_a='pos')
    except Exception:
        return np.linalg.inv(mat)

class BaseSolver:
    """Base class for solvers."""
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual):
        self.realized_effects = realized_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m
        self.V_op = VLinearOperator(self.realized_effects, self.realized_residual)

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class IterativeSolver(BaseSolver):
    """Iterative Conjugate Gradient Solver."""
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual, preconditioner: bool = True, cg_maxiter: int = 1000):
        super().__init__(realized_effects, realized_residual)
        self.use_preconditioner = preconditioner
        self.cg_maxiter = cg_maxiter
        
        self.M_op = None
        
        if self.use_preconditioner:
            R = self.realized_residual.term.cov
            # If the matrix is strictly singular, let it surface the error (consistent with WoodburySolver) and ask to disable preconditioning
            R_inv = _invert_matrix(R)
            self.M_op = ResidualPreconditioner(R_inv, self.n, self.m)

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        if marginal_residual.ndim == 2:
            K = marginal_residual.shape[1]
            prec_resid = np.empty_like(marginal_residual)
            for i in range(K):
                sol, info = cg(A=self.V_op, b=marginal_residual[:, i], M=self.M_op, maxiter=self.cg_maxiter)
                if info < 0:
                    print(f"Warning: CG info={info}")
                prec_resid[:, i] = sol
        else:
            prec_resid, info = cg(A=self.V_op, b=marginal_residual, M=self.M_op, maxiter=self.cg_maxiter)
            if info < 0:
                print(f"Warning: CG info={info}")
        
        return prec_resid

class WoodburySolver(BaseSolver):
    """Woodbury Matrix Identity Direct Solver."""
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual):
        super().__init__(realized_effects, realized_residual)

        R = self.realized_residual.term.cov
        self.R_inv = _invert_matrix(R)
        
        # Construct S matrix once
        S_blocks = []
        for i, re_i in enumerate(self.realized_effects):
            row_blocks = []
            for j, re_j in enumerate(self.realized_effects):
                Z_i_T_Z_j = re_i.Z.T @ re_j.Z
                S_ij = sparse.kron(self.R_inv, Z_i_T_Z_j)
                
                if i == j:
                    D_inv = _invert_matrix(re_i.term.cov)
                    I_oi = sparse.eye(re_i.o)
                    C_inv_ii = sparse.kron(D_inv, I_oi)
                    S_ij = S_ij + C_inv_ii
                    
                row_blocks.append(S_ij)
            S_blocks.append(row_blocks)
            
        if S_blocks:
            self.S = sparse.bmat(S_blocks, format='csc')
        else:
            self.S = None

    def _apply_R_inv_kron(self, x: np.ndarray) -> np.ndarray:
        r"""Computes (R^{-1} \otimes I_n) x efficiently for 1D or 2D arrays."""
        is_2d = x.ndim == 2
        K = x.shape[1] if is_2d else 1
        
        if is_2d:  # multiple independent RHS columns not a 2D marginal residual vector
            x_mat = x.reshape((self.m, self.n * K))
            return (self.R_inv @ x_mat).reshape((self.m, self.n, K)).reshape(self.m * self.n, K)
        else:
            x_mat = x.reshape((self.m, self.n))
            return (self.R_inv @ x_mat).ravel()

    def solve(self, marginal_residual: np.ndarray) -> np.ndarray:
        m = self.m
        n = self.n
        
        is_2d = marginal_residual.ndim == 2
        K = marginal_residual.shape[1] if is_2d else 1

        # 1. Compute A^{-1} x = (R^{-1} \otimes I_n) x
        A_inv_x = self._apply_R_inv_kron(marginal_residual)
        
        if self.S is not None:
            # Construct v1 = Z^T A^{-1} x
            v1_list = []
            for re_i in self.realized_effects:
                if is_2d:  # multiple independent RHS columns not a 2D marginal residual vector
                    Z_T_blk = sparse.kron(sparse.eye(m), re_i.Z.T)
                    v1_list.append(Z_T_blk @ A_inv_x)
                else:
                    v1_list.append(re_i._kronZ_T_matvec(A_inv_x))
            
            if is_2d:  # multiple independent RHS columns not a 2D marginal residual vector
                v1 = np.vstack(v1_list)
            else:
                v1 = np.concatenate(v1_list)
            
            # 3. Solve S v2 = v1
            v2 = spsolve(self.S, v1)
            if not is_2d and v2.ndim == 2 and v2.shape[1] == 1:
                v2 = v2.ravel()
            
            # 4. Compute v3 = Z v2
            if is_2d:  # multiple independent RHS columns not a 2D marginal residual vector
                v3 = np.zeros((m * n, K))
                offset = 0
                for re_i in self.realized_effects:
                    size_i = m * re_i.q * re_i.o
                    v2_i = v2[offset:offset+size_i]
                    Z_blk = sparse.kron(sparse.eye(m), re_i.Z)
                    v3 += Z_blk @ v2_i
                    offset += size_i
            else:
                v3 = np.zeros(m * n)
                offset = 0
                for re_i in self.realized_effects:
                    size_i = m * re_i.q * re_i.o
                    v2_i = v2[offset:offset+size_i]
                    v3 += re_i._kronZ_matvec(v2_i)
                    offset += size_i
                
            # 5. Compute v4 = A^{-1} v3
            v4 = self._apply_R_inv_kron(v3)
        
            prec_resid = A_inv_x - v4
        else:
            prec_resid = A_inv_x

        return prec_resid

def build_solver(realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual, preconditioner: bool = True, cg_maxiter: int = 1000) -> BaseSolver:
    """Builds and returns the appropriate solver."""
    m = realized_residual.m
    n = realized_residual.n
    inner_dim = sum([re.o * re.q * m for re in realized_effects])
    if inner_dim < m * n:
        return WoodburySolver(realized_effects, realized_residual)
    else:
        return IterativeSolver(realized_effects, realized_residual, preconditioner, cg_maxiter)

