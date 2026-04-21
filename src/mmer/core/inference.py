import numpy as np
from .solver import build_solver
from .terms import RealizedRandomEffect, RealizedResidual

def aggregate_random_effects(prec_resid: np.ndarray, realized_effects: tuple[RealizedRandomEffect]) -> tuple:
    """
    Aggregate posterior mean random effects across all terms.
    
    Parameters
    ----------
    prec_resid : np.ndarray
        Preconditioned marginal residual vector.
    realized_effects : tuple of RealizedRandomEffect
        Realized random effect objects.

    Returns
    -------
    total_random_effect : np.ndarray
        Sum of random effects mapped to observation space.
    mu : tuple of np.ndarray
        Posterior means for each random effect block.
    """
    total_random_effect = np.zeros_like(prec_resid)
    mu = []
    for re in realized_effects:
        val = re._compute_mu(prec_resid)
        mu.append(val)
        total_random_effect += re._map_mu(val)
    return total_random_effect, tuple(mu)

def compute_random_effects_posterior(realized_effects: tuple[RealizedRandomEffect],
                                     realized_residual: RealizedResidual,
                                     y: np.ndarray, fe_predictions: np.ndarray,
                                     terms: tuple,
                                     preconditioner: bool = True) -> tuple:
    """
    Compute posterior mean random effects and residuals for a given dataset.
    
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
    terms : tuple of RandomEffectTerm
        Learned random effect terms.
    preconditioner : bool, default=True
        Whether to use preconditioner in solver.
    
    Returns
    -------
    residuals_2d : np.ndarray
        Final residuals after subtracting all effects, shape (n, m).
    total_effect_2d : np.ndarray
        Sum of random effects across all terms, shape (n, m).
    mu_reshaped : tuple of np.ndarray
        Posterior means for each random effect term, reshaped to (groups, m, q).
    """
    n, m = y.shape
    # Compute marginal residual (before random effects)
    marginal_resid = (y - fe_predictions).T.ravel()
    
    # Solve for random effects
    solver = build_solver(realized_effects, realized_residual, preconditioner)
    prec_resid, _, _ = solver.solve(marginal_resid)
    
    # Aggregate random effects
    total_random_effect, mu = aggregate_random_effects(prec_resid, realized_effects)
    
    # Compute final residuals (after subtracting random effects)
    residuals = marginal_resid - total_random_effect
    
    residuals_2d = residuals.reshape((m, -1)).T
    total_effect_2d = total_random_effect.reshape((m, -1)).T

    mu_reshaped = []
    for k, mu_k in enumerate(mu):
        q = terms[k].q
        mu_reshaped.append(mu_k.reshape((m, q, -1)).transpose(2, 0, 1))
        
    return residuals_2d, total_effect_2d, tuple(mu_reshaped)
