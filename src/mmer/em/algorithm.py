import sys
import time
import traceback
import warnings
import numpy as np
from ..linalg.solver import build_solver
from ..structures.inference import aggregate_random_effects
from ..structures.covariance import RandomCovariance, ResidualCovariance
from ..structures.design import RandomDesign


class EMSolver:
    """
    Transient EM Solver.
    Encapsulates memory buffers, pure in-place updates, and coordinates
    the Expectation-Maximization loop cleanly.
    """

    def __init__(
        self,
        fixed_effects_model,
        max_iter,
        convergence_monitor,
        variance_corrector,
        preconditioner,
        cg_maxiter,
        n_responses,
        n_probes,
        slq_steps,
        n_jobs,
        backend,
        force_iterative=False,
    ):
        self.fixed_effects_model = fixed_effects_model
        self.max_iter = max_iter
        self.convergence_monitor = convergence_monitor
        self.variance_corrector = variance_corrector
        self.preconditioner = preconditioner
        self.cg_maxiter = cg_maxiter
        self.n_responses = n_responses
        self.n_probes = n_probes
        self.slq_steps = slq_steps
        self.n_jobs = n_jobs
        self.backend = backend
        self.force_iterative = force_iterative

    def _alloc(self, X: np.ndarray, y: np.ndarray):
        self.n_samples = X.shape[0]

        # Buffers for zero-allocation EM loop updates
        self._y_adj_C = np.empty((self.n_samples, self.n_responses), order="C")
        self._marginal_resid_buf = np.empty(
            (self.n_samples * self.n_responses,), order="C"
        )

        self._eps_2d = np.empty((self.n_responses, self.n_samples), order="C")
        self._eps_flat = self._eps_2d.ravel()

        self._T_sum = np.empty((self.n_responses, self.n_responses), order="C")
        self._eps_cross = np.empty((self.n_responses, self.n_responses), order="C")

        self._mu_cross_bufs = [
            np.empty((cov.n_responses * cov.n_effects, cov.n_responses * cov.n_effects))
            for cov in self.random_covs
        ]

    def run(
        self,
        X,
        y,
        random_covs: tuple[RandomCovariance, ...],
        random_designs: tuple[RandomDesign, ...],
        resid_cov: ResidualCovariance,
    ):
        self.random_covs = random_covs
        self.random_designs = random_designs
        self.resid_cov = resid_cov

        self.convergence_monitor.reset()
        self._alloc(X, y)

        # Initial fit with zero random effects
        self._fit_fixed_effects(X, y, np.zeros(self.n_samples * self.n_responses))
        marginal_residual = self._evaluate_residuals(X, y)

        _t0 = time.monotonic()
        _bar_width = 30
        _status = "Fitting ..."

        def _progress(iteration, status):
            filled = int(_bar_width * iteration / self.max_iter)
            bar = "█" * filled + "░" * (_bar_width - filled)
            elapsed = time.monotonic() - _t0
            line = f"\rMMER [{bar}] {iteration}/{self.max_iter}  {elapsed:.1f}s  {status}   "
            sys.stderr.write(line)
            sys.stderr.flush()

        for iteration in range(1, self.max_iter + 1):
            _progress(iteration, _status)
            marginal_residual = self._run_em_iter(X, y, marginal_residual, iteration)

            if self.convergence_monitor.is_converged:
                if self.convergence_monitor.is_early_stopped:
                    self.convergence_monitor.restore_best_state(self)
                    if np.isinf(self.convergence_monitor.log_likelihood[-1]):
                        _status = "Finished: numerical limits reached!"
                    else:
                        _status = "Finished: no further improvement!"
                else:
                    _status = "Converged: tolerance reached!"
                _progress(iteration, _status)
                break

        sys.stderr.write("\n")
        sys.stderr.flush()

    def _run_em_iter(self, X, y, marginal_residual, iteration):
        total_random_effects, mu, solver = self._e_step(marginal_residual)

        if self.convergence_monitor.is_converged:
            return marginal_residual

        try:
            self._fit_fixed_effects(X, y, total_random_effects)
            marginal_residual = self._evaluate_residuals(X, y)

            self._m_step(
                marginal_residual,
                total_random_effects,
                mu,
                solver,
                iteration,
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
            print(f"Exception in M-step: {e}")
            # traceback is imported at the top of the module — no deferred import.
            traceback.print_exc()
            warnings.warn(
                "Numerical instability encountered during M-step. Reverting to the best valid state.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.convergence_monitor.update(-np.inf, self)

        return marginal_residual

    def _e_step(self, marginal_residual):
        try:
            solver = build_solver(
                self.random_covs,
                self.random_designs,
                self.resid_cov,
                self.n_samples,
                self.preconditioner,
                self.cg_maxiter,
                force_iterative=self.force_iterative,
            )

            self.force_iterative = solver.is_iterative
            preconditioned_residuals = solver.solve(marginal_residual)

            current_log_lh = self._log_likelihood(
                marginal_residual, preconditioned_residuals, solver
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            warnings.warn(
                "Numerical instability encountered during E-step. Reverting to the best valid state.",
                RuntimeWarning,
                stacklevel=2,
            )
            current_log_lh = -np.inf
            solver = None

        self.convergence_monitor.update(current_log_lh, self)

        if self.convergence_monitor.is_converged:
            return None, None, None

        total_random_effects, mu = aggregate_random_effects(
            preconditioned_residuals,
            self.random_covs,
            self.random_designs,
            self.n_samples,
        )
        return total_random_effects, mu, solver

    def _m_step(
        self,
        marginal_residual: np.ndarray,
        total_random_effects: np.ndarray,
        mu: tuple[np.ndarray, ...],
        solver,
        iteration: int = 0,
    ):
        np.subtract(marginal_residual, total_random_effects, out=self._eps_flat)

        self._T_sum.fill(0.0)

        for k, (cov, d) in enumerate(zip(self.random_covs, self.random_designs)):
            T_k, W_k = self.variance_corrector.compute_correction(
                k, solver, n_probes=self.n_probes, iteration=iteration
            )
            self._T_sum += T_k

            mu_k_flat = mu[k].reshape((cov.n_responses * cov.n_effects, d.n_levels))
            np.dot(mu_k_flat, mu_k_flat.T, out=self._mu_cross_bufs[k])

            mu_cross = self._mu_cross_bufs[k]  # (mq, mq)
            new_cov = mu_cross + W_k
            new_cov /= d.n_levels

            self.random_covs[k].update(new_cov)

        np.dot(self._eps_2d, self._eps_2d.T, out=self._eps_cross)

        self._eps_cross += self._T_sum
        self._eps_cross /= self.n_samples

        self.resid_cov.update(self._eps_cross)
        return self

    def _fit_fixed_effects(
        self, X: np.ndarray, y: np.ndarray, total_random_effects: np.ndarray
    ):
        """Train fixed-effect component in isolation (SoC)"""

        total_re_2d = total_random_effects.reshape((self.n_responses, self.n_samples)).T
        np.subtract(y, total_re_2d, out=self._y_adj_C)

        y_fit = self._y_adj_C if self.n_responses != 1 else self._y_adj_C.ravel()
        self.fixed_effects_model.fit(X, y_fit)

    def _evaluate_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        fx = self.fixed_effects_model.predict(X)
        if self.n_responses == 1 and fx.ndim == 1:
            fx = fx[:, None]
        np.subtract(y, fx, out=self._y_adj_C)
        self._marginal_resid_buf.reshape((self.n_responses, self.n_samples))[:] = (
            self._y_adj_C.T
        )
        return self._marginal_resid_buf

    def _log_likelihood(self, marginal_residual, preconditioned_residuals, solver):
        log_det_V = solver.logdet(
            slq_steps=self.slq_steps,
            n_probes=self.n_probes,
            n_jobs=self.n_jobs,
            backend=self.backend,
        )
        return -0.5 * (
            self.n_responses * self.n_samples * np.log(2 * np.pi)
            + log_det_V
            + np.dot(marginal_residual, preconditioned_residuals)
        )
