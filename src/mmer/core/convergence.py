import copy
import numpy as np

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
    """
    def __init__(self, tol: float = 1e-6, patience: int = 3):
        self.tol = tol
        self.patience = max(1, patience)
        self.reset()
        
    def reset(self):
        """Reset the convergence state for a new fitting run."""
        self.log_likelihood = []
        self.is_converged = False
        self._best_log_likelihood = -np.inf
        self._no_improvement_count = 0
        self._best_state = None
    
    def update(self, current_log_likelihood: float, current_state: dict) -> bool:
        """
        Update convergence monitor with new log-likelihood value.
        
        Checks both relative change tolerance and patience-based early stopping.
        Stores the best state encountered during optimization.
        
        Parameters
        ----------
        current_log_likelihood : float
            Current iteration's log-likelihood value.
        current_state : dict
            Current model state containing 're_covs', 'resid_cov', and 'fe_model'
            to save if this is the best state so far.
        
        Returns
        -------
        is_converged : bool
            Whether model has converged based on tolerance or patience.
        """
        self.log_likelihood.append(current_log_likelihood)
        
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
                key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
                for key, value in current_state.items()
            }
        else:
            self._no_improvement_count += 1
        
        # Check patience stopping
        if self._no_improvement_count >= self.patience:
            self.is_converged = True
        
        return self.is_converged
