import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve
from scipy.linalg import solve
from .operator import VLinearOperator, ResidualPreconditioner
from .terms import RealizedRandomEffect, RealizedResidual

class BaseSolver:
    """Base class for solvers."""
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual):
        self.realized_effects = realized_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m

    def solve(self, marginal_residual: np.ndarray) -> tuple:
        raise NotImplementedError
        
class IterativeSolver(BaseSolver):
    """Iterative Conjugate Gradient Solver."""
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual, preconditioner: bool = True, cg_maxiter: int = 1000):
        super().__init__(realized_effects, realized_residual)
        self.use_preconditioner = preconditioner
        self.cg_maxiter = cg_maxiter
        
        self.V_op = VLinearOperator(self.realized_effects, self.realized_residual)
        self.M_op = None
        
        if self.use_preconditioner:
            try:
                resid_cov_inv = solve(a=self.realized_residual.cov, b=np.eye(self.m), assume_a='pos')
                self.M_op = ResidualPreconditioner(resid_cov_inv, self.n, self.m)
            except Exception:
                pass

    def solve(self, marginal_residual: np.ndarray) -> tuple:
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
        
        return prec_resid, self.V_op, self.M_op

class WoodburySolver(BaseSolver):
    """Woodbury Matrix Identity Direct Solver."""
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual):
        super().__init__(realized_effects, realized_residual)
        
        m = self.m
        R = getattr(self.realized_residual, 'cov', getattr(self.realized_residual.term, 'cov', None))
        self.R_inv = np.linalg.inv(R)
        
        # Construct S matrix once
        S_blocks = []
        for i, re_i in enumerate(self.realized_effects):
            row_blocks = []
            for j, re_j in enumerate(self.realized_effects):
                Z_i_T_Z_j = re_i.Z.T @ re_j.Z
                S_ij = sparse.kron(self.R_inv, Z_i_T_Z_j)
                
                if i == j:
                    D_inv = np.linalg.inv(re_i.term.cov)
                    I_oi = sparse.eye(re_i.o)
                    C_inv_ii = sparse.kron(D_inv, I_oi)
                    S_ij = S_ij + C_inv_ii
                    
                row_blocks.append(S_ij)
            S_blocks.append(row_blocks)
            
        if S_blocks:
            self.S = sparse.bmat(S_blocks, format='csc')
        else:
            self.S = None

        self.V_op = VLinearOperator(self.realized_effects, self.realized_residual)

    def solve(self, marginal_residual: np.ndarray) -> tuple:
        m = self.m
        n = self.n
        
        is_2d = marginal_residual.ndim == 2
        K = marginal_residual.shape[1] if is_2d else 1
        
        # 1. Compute A^{-1} x = (R^{-1} \otimes I_n) x
        if is_2d:
            x_mat = marginal_residual.reshape((m, n * K))
            A_inv_x = (self.R_inv @ x_mat).reshape((m, n, K)).reshape(m * n, K)
        else:
            x_mat = marginal_residual.reshape((m, n))
            A_inv_x = (self.R_inv @ x_mat).ravel()
        
        if self.S is not None:
            # Construct v1 = Z^T A^{-1} x
            v1_list = []
            for re_i in self.realized_effects:
                if is_2d:
                    Z_T_blk = sparse.kron(sparse.eye(m), re_i.Z.T)
                    v1_list.append(Z_T_blk @ A_inv_x)
                else:
                    v1_list.append(re_i._kronZ_T_matvec(A_inv_x))
            
            if is_2d:
                v1 = np.vstack(v1_list)
            else:
                v1 = np.concatenate(v1_list)
            
            # 3. Solve S v2 = v1
            v2 = spsolve(self.S, v1)
            if not is_2d and v2.ndim == 2 and v2.shape[1] == 1:
                v2 = v2.ravel()
            
            # 4. Compute v3 = Z v2
            if is_2d:
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
            if is_2d:
                v3_mat = v3.reshape((m, n * K))
                v4 = (self.R_inv @ v3_mat).reshape((m, n, K)).reshape(m * n, K)
            else:
                v3_mat = v3.reshape((m, n))
                v4 = (self.R_inv @ v3_mat).ravel()
            
            prec_resid = A_inv_x - v4
        else:
            prec_resid = A_inv_x
            
        return prec_resid, self.V_op, None

def build_solver(realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual, preconditioner: bool = True, cg_maxiter: int = 1000) -> BaseSolver:
    """Builds and returns the appropriate solver."""
    m = realized_residual.m
    n = realized_residual.n
    inner_dim = sum([re.o * re.q * m for re in realized_effects])
    if inner_dim < m * n:
        return WoodburySolver(realized_effects, realized_residual)
    else:
        return IterativeSolver(realized_effects, realized_residual, preconditioner, cg_maxiter)

