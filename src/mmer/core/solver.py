import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import solve
from .operator import VLinearOperator, ResidualPreconditioner
from .terms import RealizedRandomEffect, RealizedResidual

class SolverContext:
    """
    Encapsulates solver setup and execution.
    
    Handles preconditioner creation, VLinearOperator setup, and CG invocation
    in one place to eliminate duplicated solver code.
    
    Parameters
    ----------
    realized_effects : tuple of RealizedRandomEffect
        Realized random effects.
    realized_residual : RealizedResidual
        Realized residual term (for preconditioner).
    preconditioner : bool, default=True
        Whether to use preconditioner.
    
    Attributes
    ----------
    realized_effects : tuple of RealizedRandomEffect
        Stored realized random effects.
    realized_residual : RealizedResidual
        Stored realized residual term.
    n : int
        Dataset size, extracted from realized_residual.
    m : int
        Number of outputs, extracted from realized_residual.
    use_preconditioner : bool
        Whether preconditioner will be applied.
    """
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual, preconditioner: bool = True):
        self.realized_effects = realized_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m
        self.use_preconditioner = preconditioner
    
    def solve(self, marginal_residual: np.ndarray) -> tuple:
        """
        Solve V * x = marginal_residual using conjugate gradient.
        
        Parameters
        ----------
        marginal_residual : np.ndarray
            Right-hand side of linear system, shape (m*n,).
        
        Returns
        -------
        prec_resid : np.ndarray
            Solution vector, shape (m*n,).
        V_op : VLinearOperator
            Linear operator used in solve.
        M_op : ResidualPreconditioner or None
            Preconditioner used (if any).
        """
        V_op = VLinearOperator(self.realized_effects, self.realized_residual)
        M_op = None
        
        if self.use_preconditioner:
            try:
                resid_cov_inv = solve(a=self.realized_residual.cov, b=np.eye(self.m), assume_a='pos')
                M_op = ResidualPreconditioner(resid_cov_inv, self.n, self.m)
            except Exception:
                pass
        
        prec_resid, info = cg(A=V_op, b=marginal_residual, M=M_op)
        # Todo: Using maxiter and tol to control convergence and avoid infinite loops.
        if info != 0:
            print(f"Warning: CG info={info}")
        
        return prec_resid, V_op, M_op
