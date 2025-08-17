import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import solve
from sklearn.base import RegressorMixin
from tqdm import tqdm
from .operator import VLinearOperator, ResidualPreconditioner, compute_cov_correction
from ..lanczos_algorithm import slq
from .random_effect import RandomEffect
from .residual import Residual
from .mixed_result import MixedEffectResults

class MixedEffectRegressor:
    """
    Multivariate Mixed Effects Regression.
    It supports multiple responses, any fixed effects model, multiple random effects, and multiple grouping factors.
    Parameters:
        fixed_effects_model: A regressor with fit and predict methods that supports multi-output regression.
        max_iter: Maximum number of EM iterations.
        tol: Log-likelihood convergence tolerance.
        slq_steps: Number of steps for SLQ approximation.
        slq_probes: Number of probes for SLQ approximation.
        preconditioner: Whether to use a preconditioner for CG solver with marginal covariance.
        correction_method: Method for covariance correction, options are 'ste', 'bste', or 'detr'.
        n_jobs: Number of parallel jobs for SLQ computation and covariance correction.
        backend: Backend for parallel processing, options are 'loky' or 'threading'.
    """
    _VALID_CORRECTION_METHODS = ['ste', 'bste', 'detr', 'xbste']
    _VALID_CONVERGENCE_CRITERIA = ['norm', 'log_lh']

    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 10, tol: float = 1e-3,
                 slq_steps: int = 30, slq_probes: int = 30, preconditioner: bool = True, correction_method: str = 'bste',
                 convergence_criterion: str = 'norm', n_jobs: int = 1, backend: str = 'loky'):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.slq_steps = slq_steps
        self.slq_probes = slq_probes
        self.preconditioner = preconditioner

        self.correction_method = correction_method
        if self.correction_method not in self._VALID_CORRECTION_METHODS:
            raise ValueError(f"Unknown correction method: '{self.correction_method}'. Available methods are {self._VALID_CORRECTION_METHODS}.")
        
        self.convergence_criterion = convergence_criterion
        if self.convergence_criterion not in self._VALID_CONVERGENCE_CRITERIA:
            raise ValueError(f"convergence_criterion must be one of {self._VALID_CONVERGENCE_CRITERIA}")

        self.log_likelihood = []
        self.track_change = []
        self._is_converged = False
        self.n_jobs = n_jobs
        self.backend = backend

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | tuple[list[int] | None]):
        """
        Prepare initial parameters, instances of random effects and residuals.
        """
        self.n, self.m = y.shape
        self.k = groups.shape[1]
        if random_slopes is None:
            self.random_slopes = tuple(None for _ in range(self.k))
        elif len(random_slopes) != self.k:
            raise ValueError(f"Length of random_slopes tuple ({len(random_slopes)}) " f"must match the number of groups ({self.k}).")
        else:
            self.random_slopes = random_slopes

        random_effects = tuple(RandomEffect(self.n, self.m, k, slope_col) for k, slope_col in enumerate(self.random_slopes))

        for re in random_effects:
            re.design_random_effect(X, groups)
        residual = Residual(self.n, self.m)

        marginal_residual = self.compute_marginal_residual(X, y, 0.0)

        return marginal_residual, random_effects, residual
    
    def compute_marginal_residual(self, X: np.ndarray, y: np.ndarray, total_random_effect: np.ndarray):
        """
        Compute marginal residuals by fitting the fixed effects models to the adjusted response variables.
        returns:
            1d array (mn,)
        """
        y_adj = y - total_random_effect
        y_adj = y_adj.ravel() if self.m == 1 else y_adj
        fx = self.fe_model.fit(X, y_adj).predict(X)
        fx = fx[:, None] if self.m == 1 else fx
        return (y - fx).T.ravel()

    def compute_log_likelihood(self, marginal_residual: np.ndarray, prec_resid: np.ndarray, V_op: VLinearOperator):
        """
        Compute the log-likelihood of the marginal distribution of the residuals.
        aka the marginal log-likelihood
            marginal_residual: marginal residuals y-fx
            prec_resid: precision-weighted residuals V⁻¹(y-fx)
        """
        log_det_V = slq.logdet(V_op, self.slq_steps, self.slq_probes, self.n_jobs, self.backend)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + marginal_residual.T @ prec_resid) / 2
        return log_likelihood

    def aggregate_random_effects(self, prec_resid: np.ndarray, random_effects: tuple[RandomEffect, ...]):
        """
        Computes sum of all random effects in observation space.
            Σₖ(Iₘ ⊗ Zₖ)μₖ
        returns:
            1d array (mn,) and tuple of mu
        """
        total_random_effect = np.zeros(self.m * self.n)
        mu = []
        for re in random_effects:
            mu.append(re.compute_mu(prec_resid))
            total_random_effect += re.map_mu(mu[-1])
        return total_random_effect, tuple(mu)
    
    def _solver(self, marginal_residual: np.ndarray, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Solves the system of equations to obtain the precision-weighted residuals.
        """
        V_op = VLinearOperator(random_effects, residual)
        M_op = None
        if self.preconditioner:
            try:
                resid_cov_inv = solve(a=residual.cov, b=np.eye(self.m), assume_a='pos')
                M_op = ResidualPreconditioner(resid_cov_inv, self.n, self.m)
            except Exception:
                print("Warning: Singular residual covariance. If the fixed-effects model absorbs nearly all degrees of freedom, residual variance may vanish, leading to singularity.")
            
        prec_resid, info = cg(A=V_op, b=marginal_residual, M=M_op)
        if info != 0:
            print(f"Warning: CG solver (V⁻¹(y-fx)) did not converge. Info={info}")
        return prec_resid, V_op, M_op

    def _e_step(self, marginal_residual: np.ndarray, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Performs the E-step of the EM algorithm.
        """
        prec_resid, V_op, M_op = self._solver(marginal_residual, random_effects, residual)
        if self.convergence_criterion == 'log_lh':
            final_logL = self.compute_log_likelihood(marginal_residual, prec_resid, V_op)
            self.log_likelihood.append(final_logL)
        total_random_effect, mu = self.aggregate_random_effects(prec_resid, random_effects)

        return total_random_effect, mu, V_op, M_op

    def _m_step(self, marginal_residual: np.ndarray, total_random_effect: np.ndarray, mu: tuple[np.ndarray],
                random_effects: tuple[RandomEffect, ...], residual: Residual, V_op: VLinearOperator, M_op: ResidualPreconditioner):
        """
        Performs the M-step of the EM algorithm.
        """
        eps = marginal_residual - total_random_effect
        T_sum = np.zeros((self.m, self.m))
        new_tau = []
        for k, re in enumerate(random_effects):
            T_k, W_k = compute_cov_correction(k, V_op, M_op, self.correction_method, self.n_jobs, self.backend)
            T_sum += T_k
            new_tau.append(re.compute_cov(mu[k], W_k))

        residual.cov[...] = residual.compute_cov(eps, T_sum)
        for k, re in enumerate(random_effects):
            re.cov[...] = new_tau[k]

        return self

    def _check_convergence(self, old_phi: np.ndarray, old_tau: tuple[np.ndarray], random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Checks if the model parameters have converged.
        """
        if self.convergence_criterion == 'log_lh':
            change = np.abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2])
            self.track_change.append(change)
            self._is_converged = change < self.tol
            if self.log_likelihood[-1] <= self.log_likelihood[-2]:
                self._is_converged = True
        elif self.convergence_criterion == 'norm':
            param_changes  = [np.linalg.norm(re.cov - tau_k) / np.linalg.norm(tau_k)
                              for re, tau_k in zip(random_effects, old_tau)]
            param_changes.append(np.linalg.norm(residual.cov - old_phi) / np.linalg.norm(old_phi))
            change = np.max(param_changes)
            self.track_change.append(change)
            self._is_converged = change < self.tol
        return self

    def _run_em_iteration(self, X: np.ndarray, y: np.ndarray, marginal_residual: np.ndarray, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Run the EM algorithm for fitting the model.
        """
        total_random_effect, mu, V_op, M_op = self._e_step(marginal_residual, random_effects, residual)

        marginal_residual[...] = self.compute_marginal_residual(X, y, total_random_effect.reshape((self.m, self.n)).T)

        self._m_step(marginal_residual, total_random_effect, mu, random_effects, residual, V_op, M_op)

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | tuple[list[int]] = None):
        """
        Fit the mixed effects model using the EM algorithm.
        """
        return self._fit(X=X, y=y, groups=groups, random_slopes=random_slopes)
    
    def partial_fit(self, marginal_residual: np.ndarray, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Partially fit the model using the previously fitted model parameters.
        """
        return self._fit(marginal_residual=marginal_residual, random_effects=random_effects, residual=residual)

    def _fit(self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None, random_slopes: None | tuple[list[int]] = None,
             marginal_residual: np.ndarray = None, random_effects: tuple[RandomEffect, ...] = None, residual: Residual = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.
        It can also be used for partial fitting to continue training for more iterations.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slopes: tuple[list[int]] dictionary mapping group indices to lists of random slope indices (optional).

            marginal_residual: (n_samples, M) array of M marginal residuals (optional).
            random_effects: tuple[RandomEffect, ...] random effects (optional).
            residual: Residual object containing residual information (optional).

        Returns:
            MixedEffectResults: Contains fitted model and results.
        """
        if marginal_residual is None:
            marginal_residual, random_effects, residual = self.prepare_data(X, y, groups, random_slopes)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting Model",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            if self.convergence_criterion == 'norm':
                old_tau = tuple(re.cov.copy() for re in random_effects)
                old_phi = residual.cov.copy()
            else:
                old_tau = None
                old_phi = None

            self._run_em_iteration(X, y, marginal_residual, random_effects, residual)
            if iter_ > 2:
                self._check_convergence(old_phi, old_tau, random_effects, residual)
                if self._is_converged:
                    pbar.set_description("Model Converged")
                    break
        # Final log-likelihood calculation
        prec_resid, V_op, M_op = self._solver(marginal_residual, random_effects, residual)
        self.log_likelihood.append(self.compute_log_likelihood(marginal_residual, prec_resid, V_op))

        return MixedEffectResults(self, random_effects, residual)
