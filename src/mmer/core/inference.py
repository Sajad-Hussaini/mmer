import numpy as np
from .solver import SolverContext
from .terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual

class InferenceEngine:
    """
    Handles post-fit inference computations for random effects and residuals.
    
    Decouples inference logic from the EM fitting loop, allowing reuse across
    compute_random_effects() and enhanced predict() methods. Computes posterior
    means of random effects given fitted model parameters.
    
    Parameters
    ----------
    random_effect_terms : tuple of RandomEffectTerm
        Learned random effect terms (fitted state).
    residual_term : ResidualTerm
        Learned residual term (fitted state).
    n : int
        Dataset size.
    preconditioner : bool, default=True
        Whether to use preconditioner in solver.
    
    Attributes
    ----------
    random_effect_terms : tuple of RandomEffectTerm
        Stored random effect terms.
    residual_term : ResidualTerm
        Stored residual term.
    n : int
        Dataset size.
    m : int
        Number of outputs, extracted from residual_term.
    preconditioner : bool
        Whether to use preconditioner.
    """
    def __init__(self, random_effect_terms: tuple[RandomEffectTerm], residual_term: ResidualTerm, n: int, preconditioner: bool = True):
        self.random_effect_terms = random_effect_terms
        self.residual_term = residual_term
        self.n = n
        self.m = residual_term.m
        self.preconditioner = preconditioner
    
    def compute_random_effects(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual,
                               y: np.ndarray, fe_predictions: np.ndarray) -> tuple:
        """
        Compute posterior mean random effects and residuals.
        
        Given observations and fixed effect predictions, computes the posterior
        mean of random effects for each grouping factor and the final residuals.
        
        Parameters
        ----------
        realized_effects : tuple of RealizedRandomEffect
            Realized random effect objects for current data.
        realized_residual : RealizedResidual
            Realized residual term for current data.
        y : np.ndarray
            Target values, shape (n, m).
        fe_predictions : np.ndarray
            Fixed effect predictions, shape (n, m).
        
        Returns
        -------
        residuals : np.ndarray
            Final residuals after subtracting all effects, raveled shape (m*n,).
        random_effects_sum : np.ndarray
            Sum of random effects across all terms, raveled shape (m*n,).
        mu : tuple of np.ndarray
            Posterior means for each random effect term.
        """
        # Compute marginal residual (before random effects)
        marginal_resid = (y - fe_predictions).T.ravel()
        
        # Solve for random effects
        solver_ctx = SolverContext(realized_effects, realized_residual, self.preconditioner)
        prec_resid, _, _ = solver_ctx.solve(marginal_resid)
        
        # Aggregate random effects
        total_random_effect = np.zeros(self.m * self.n)
        mu = []
        for re in realized_effects:
            val = re._compute_mu(prec_resid)
            mu.append(val)
            total_random_effect += re._map_mu(val)
        
        # Compute final residuals (after subtracting random effects)
        residuals = marginal_resid - total_random_effect
        
        return residuals, total_random_effect, tuple(mu)
