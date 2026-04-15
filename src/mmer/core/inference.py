from __future__ import annotations

import numpy as np

from .solver import SolverContext
from .terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual


class InferenceEngine:
    """
    Compute posterior mean random effects and residuals after fitting.

    This keeps inference separate from fitting so it can be reused by the
    estimator and the result object without duplicating the solve logic.
    """

    def __init__(
        self,
        random_effect_terms: tuple[RandomEffectTerm, ...],
        residual_term: ResidualTerm,
        n: int,
        preconditioner: bool = True,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
    ):
        self.random_effect_terms = random_effect_terms
        self.residual_term = residual_term
        self.n = n
        self.m = residual_term.m
        self.preconditioner = preconditioner
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter

    def compute_random_effects(
        self,
        realized_effects: tuple[RealizedRandomEffect, ...],
        realized_residual: RealizedResidual,
        y: np.ndarray,
        fe_predictions: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> tuple:
        marginal_resid = (y - fe_predictions).T.ravel()

        solver_ctx = SolverContext(
            realized_effects,
            realized_residual,
            self.preconditioner,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
        )
        prec_resid, _, _ = solver_ctx.solve(marginal_resid, x0=x0)

        total_random_effect = np.zeros(self.m * self.n)
        mu = []
        for re in realized_effects:
            val = re._compute_mu(prec_resid)
            mu.append(val)
            total_random_effect += re._map_mu(val)

        residuals = marginal_resid - total_random_effect
        return residuals, total_random_effect, tuple(mu), prec_resid
