import numpy as np


class ConvergenceMonitor:
    """
    Track EM convergence using only scalar diagnostics.

    The monitor keeps the stopping logic separate from the model state so
    iterations do not spend time copying large arrays.
    """

    def __init__(self, tol: float = 1e-6, patience: int = 3):
        self.tol = tol
        self.patience = max(1, patience)
        self.reset()

    def reset(self):
        self.log_likelihood = []
        self.is_converged = False
        self._best_log_likelihood = -np.inf
        self._no_improvement_count = 0

    def update(self, current_log_likelihood: float) -> bool:
        self.log_likelihood.append(current_log_likelihood)

        if len(self.log_likelihood) >= 2:
            previous = self.log_likelihood[-2]
            denominator = max(np.abs(previous), np.finfo(float).eps)
            change = np.abs((self.log_likelihood[-1] - previous) / denominator)
            if change <= self.tol:
                self.is_converged = True

        if current_log_likelihood > self._best_log_likelihood:
            self._best_log_likelihood = current_log_likelihood
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            self.is_converged = True

        return self.is_converged
