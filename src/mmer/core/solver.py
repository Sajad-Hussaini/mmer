from __future__ import annotations

import numpy as np

from .linear_solver import cg_solve
from .operator import ResidualPreconditioner, VLinearOperator
from .terms import RealizedRandomEffect, RealizedResidual


class SolverContext:
    """
    Build the covariance operator, optional preconditioner, and run CG.

    The context is intentionally small so the estimator can reuse the same
    solver policy while varying only the data-dependent realized terms.
    """

    def __init__(
        self,
        realized_effects: tuple[RealizedRandomEffect, ...],
        realized_residual: RealizedResidual,
        preconditioner: bool = True,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
    ):
        self.realized_effects = realized_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m
        self.use_preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self._last_solution = None

    def solve(self, marginal_residual: np.ndarray, x0: np.ndarray | None = None) -> tuple:
        V_op = VLinearOperator(self.realized_effects, self.realized_residual)
        M_op = None

        if self.use_preconditioner:
            try:
                M_op = ResidualPreconditioner(self.realized_residual.term.cov, self.n, self.m)
            except Exception:
                M_op = None

        initial_guess = x0 if x0 is not None else self._last_solution
        prec_resid = cg_solve(
            V_op,
            marginal_residual,
            M_op,
            x0=initial_guess,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
        )
        self._last_solution = prec_resid

        return prec_resid, V_op, M_op
