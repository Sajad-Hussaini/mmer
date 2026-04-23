import copy
import pickle
import numpy as np


def _copy_model(model):
    """
    Create a safe snapshot of a fitted model.

    Tries pickle round-trip first (the standard serialization path used by
    PyTorch, sklearn, and most custom parametric classes), then falls back to
    copy.deepcopy for objects that are copyable but not picklable (e.g. objects
    containing lambda functions or open file handles).
    """
    try:
        return pickle.loads(pickle.dumps(model))
    except Exception:
        return copy.deepcopy(model)

class ConvergenceMonitor:
    """
    Tracks convergence state during EM iterations.
    
    Manages log-likelihood history, patience counter, and best-state restoration.
    Supports both relative tolerance-based stopping and early stopping based on
    patience when no improvement is observed.
    
    Parameters
    ----------
    tol : float, default=1e-6
        Convergence tolerance on log-likelihood relative change.
    patience : int, default=3
        Number of iterations to wait before early stopping if no improvement.
    
    Attributes
    ----------
    tol : float
        Convergence tolerance.
    patience : int
        Patience counter threshold.
    log_likelihood : list of float
        History of log-likelihood values across iterations.
    is_converged : bool
        Whether convergence criteria have been met.
    is_early_stopped : bool
        Whether the model training was stopped early due to patience threshold.
    """
    def __init__(self, tol: float = 1e-6, patience: int = 3):
        self.tol = tol
        self.patience = max(1, patience)
        self.reset()
        
    def reset(self):
        """Reset the convergence state for a new fitting run."""
        self.log_likelihood = []
        self.is_converged = False
        self.is_early_stopped = False
        self._best_log_likelihood = -np.inf
        self._no_improvement_count = 0
        self._best_state = None
    
    def update(self, current_log_likelihood: float, model) -> "ConvergenceMonitor":
        """
        Update convergence monitor with new log-likelihood value.
        
        Checks both relative change tolerance and patience-based early stopping.
        Stores the best state encountered during optimization.
        
        Parameters
        ----------
        current_log_likelihood : float
            Current iteration's log-likelihood value.
        model : MixedEffectRegressor
            The model instance being optimized to read and save the best state from.
        
        Returns
        -------
        self : ConvergenceMonitor
            The monitor instance (for chaining or inspection).
        """
        self.log_likelihood.append(current_log_likelihood)
        
        # Abort immediately if the model step failed (e.g. non-PD covariance)
        if np.isinf(current_log_likelihood) and current_log_likelihood < 0:
            self.is_converged = True
            self.is_early_stopped = True
            return self
            
        # Check relative change convergence
        if len(self.log_likelihood) >= 2:
            prev = self.log_likelihood[-2]
            denom = max(np.abs(prev), np.finfo(float).eps)
            change = np.abs((self.log_likelihood[-1] - prev) / denom)
            if change <= self.tol:
                self.is_converged = True
        
        # Track best state
        if current_log_likelihood > self._best_log_likelihood:
            self._best_log_likelihood = current_log_likelihood
            self._no_improvement_count = 0
            self._best_state = {
                're_covs': [term.cov.copy() for term in model.random_effect_terms],
                'resid_cov': model.residual_term.cov.copy(),
                'fe_model': _copy_model(model.fe_model),
            }
        else:
            self._no_improvement_count += 1
        
        # Check patience stopping
        if self._no_improvement_count >= self.patience:
            self.is_converged = True
            self.is_early_stopped = True
        
        return self

    def restore_best_state(self, model) -> bool:
        """
        Restore model parameters to the best observed state.

        Applies the covariance matrices and fixed-effects model that produced
        the highest log-likelihood seen during the EM run back onto *model*.
        Should be called immediately after the EM loop exits.

        Parameters
        ----------
        model : MixedEffectRegressor
            The regressor whose internal state will be overwritten.

        Returns
        -------
        restored : bool
            ``True`` if a best state was available and was applied,
            ``False`` if no update had ever been recorded (should not happen
            in practice after at least one iteration).
        """
        if self._best_state is None:
            return False
        for k, cov in enumerate(self._best_state['re_covs']):
            model.random_effect_terms[k].set_cov(cov)
        model.residual_term.set_cov(self._best_state['resid_cov'])
        model.fe_model = self._best_state['fe_model']
        return True
