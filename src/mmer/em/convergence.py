import copy
import pickle
import numpy as np


def _copy_model(model):
    try:
        return copy.deepcopy(model)
    except Exception:
        return pickle.loads(pickle.dumps(model))


class ConvergenceMonitor:
    def __init__(self, tol: float = 1e-6, patience: int = 3):
        self.tol = tol
        self.patience = max(1, patience)
        self.reset()

    def reset(self):
        self.log_likelihood = []
        self.is_converged = False
        self.is_early_stopped = False
        self._best_log_likelihood = -np.inf
        self._no_improvement_count = 0
        self._best_state = None

    def update(self, current_log_likelihood: float, model) -> "ConvergenceMonitor":
        self.log_likelihood.append(current_log_likelihood)

        if np.isinf(current_log_likelihood) and current_log_likelihood < 0:
            self.is_converged = True
            self.is_early_stopped = True
            return self

        if len(self.log_likelihood) >= 2:
            prev = self.log_likelihood[-2]
            denom = max(np.abs(prev), np.finfo(float).eps)
            change = np.abs((self.log_likelihood[-1] - prev) / denom)
            if change <= self.tol:
                self.is_converged = True

        if current_log_likelihood > self._best_log_likelihood:
            self._best_log_likelihood = current_log_likelihood
            self._no_improvement_count = 0
            self._best_state = {
                "random_covs": [cov.matrix.copy() for cov in model.random_covs],
                "resid_cov": model.resid_cov.matrix.copy(),
                "fixed_effects_model": _copy_model(model.fixed_effects_model),
            }
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            self.is_converged = True
            self.is_early_stopped = True

        return self

    def restore_best_state(self, model) -> bool:
        if self._best_state is None:
            return False
        for k, mat in enumerate(self._best_state["random_covs"]):
            model.random_covs[k].update(mat)
        model.resid_cov.update(self._best_state["resid_cov"])
        model.fixed_effects_model = self._best_state["fixed_effects_model"]
        return True
