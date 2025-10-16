import pickle
import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import solve
from sklearn.base import RegressorMixin
from tqdm import tqdm
from .operator import VLinearOperator, ResidualPreconditioner, compute_cov_correction
from ..lanczos_algorithm import slq
from .random_effect import RandomEffect
from .residual import Residual


class MixedEffectRegressor:
    """
    Multivariate Mixed Effects Regression.

    Supports multiple responses, any fixed effects model, multiple random effects,
    and multiple grouping factors.

    Parameters
    ----------
    fixed_effects_model : RegressorMixin
        A regressor with fit and predict methods that supports multi-output regression.
    max_iter : int, default=10
        Maximum number of EM iterations.
    tol : float, default=1e-3
        Log-likelihood convergence tolerance.
    patience : int, default=3
        Number of iterations to wait for improvement before early stopping.
    slq_steps : int, default=30
        Number of steps for SLQ approximation.
    slq_probes : int, default=30
        Number of probes for SLQ approximation.
    preconditioner : bool, default=True
        Whether to use a preconditioner for CG solver with marginal covariance.
    correction_method : str, default='bste'
        Method for covariance correction, options are 'ste', 'bste', or 'detr'.
    n_jobs : int, default=-1
        Number of parallel jobs for SLQ computation and covariance correction.
    backend : str, default='loky'
        Backend for parallel processing, options are 'loky' or 'threading'.
    """
    _VALID_CORRECTION_METHODS = ['ste', 'bste', 'detr']

    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 10, tol: float = 1e-3, patience: int = 3,
                 slq_steps: int = 30, slq_probes: int = 30, preconditioner: bool = True, correction_method: str = 'bste',
                 n_jobs: int = -1, backend: str = 'loky'):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience if patience >= 1 else 1
        self.slq_steps = slq_steps
        self.slq_probes = slq_probes
        self.preconditioner = preconditioner

        self.correction_method = correction_method
        if self.correction_method not in self._VALID_CORRECTION_METHODS:
            raise ValueError(f"Unknown correction method: '{self.correction_method}'. Available methods are {self._VALID_CORRECTION_METHODS}.")
        
        self.n_jobs = n_jobs
        self.backend = backend

        self.log_likelihood = []
        self._is_converged = False
        self._no_improvement_count = 0
        self._best_log_likelihood = -np.inf
        self._best_re_covs = None
        self._best_resid_cov = None
        self._best_fe_model = None

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | tuple[list[int] | None]):
        """
        Prepare initial parameters, instances of random effects and residuals.

        Parameters
        ----------
        X : np.ndarray
            Fixed effect covariates of shape (n_samples, n_features).
        y : np.ndarray
            Response variables of shape (n_samples, n_outputs).
        groups : np.ndarray
            Grouping factors of shape (n_samples, n_groups).
        random_slopes : None or tuple of list of int
            Random slope column indices for each grouping factor.

        Returns
        -------
        marginal_residual : np.ndarray
            Initial marginal residuals.
        random_effects : tuple of RandomEffect
            Random effect objects for each grouping factor.
        residual : Residual
            Residual object.
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

        marginal_residual = self._compute_marginal_residual(X, y, 0.0)

        return marginal_residual, random_effects, residual
    
    def _compute_marginal_residual(self, X: np.ndarray, y: np.ndarray, total_random_effect: np.ndarray):
        """
        Compute marginal residuals by fitting the fixed effects model to the adjusted response.

        Parameters
        ----------
        X : np.ndarray
            Fixed effect covariates of shape (n_samples, n_features).
        y : np.ndarray
            Response variables of shape (n_samples, n_outputs).
        total_random_effect : np.ndarray
            Sum of all random effects.

        Returns
        -------
        np.ndarray
            Marginal residuals of shape (n_outputs * n_samples).
        """
        y_adj = y - total_random_effect
        y_adj = y_adj.ravel() if self.m == 1 else y_adj
        fx = self.fe_model.fit(X, y_adj).predict(X)
        fx = fx[:, None] if self.m == 1 else fx
        return (y - fx).T.ravel()

    def _compute_log_likelihood(self, marginal_residual: np.ndarray, prec_resid: np.ndarray, V_op: VLinearOperator):
        """
        Compute the marginal log-likelihood.

        Parameters
        ----------
        marginal_residual : np.ndarray
            Marginal residuals (y - fx).
        prec_resid : np.ndarray
            Precision-weighted residuals V⁻¹(y - fx).
        V_op : VLinearOperator
            Linear operator representing the marginal covariance V.

        Returns
        -------
        float
            The marginal log-likelihood value.
        """
        log_det_V = slq.logdet(V_op, self.slq_steps, self.slq_probes, self.n_jobs, self.backend)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + marginal_residual.T @ prec_resid) / 2
        return log_likelihood

    def _aggregate_random_effects(self, prec_resid: np.ndarray, random_effects: tuple[RandomEffect, ...]):
        """
        Compute the sum of all random effects in observation space.

        Parameters
        ----------
        prec_resid : np.ndarray
            Precision-weighted residuals V⁻¹(y - fx).
        random_effects : tuple of RandomEffect
            Random effect objects for each grouping factor.

        Returns
        -------
        total_random_effect : np.ndarray
            Sum of all random effects of shape (n_outputs * n_samples).
        mu : tuple of np.ndarray
            Random effect coefficients for each grouping factor.
        """
        total_random_effect = np.zeros(self.m * self.n)
        mu = []
        for re in random_effects:
            mu.append(re._compute_mu(prec_resid))
            total_random_effect += re._map_mu(mu[-1])
        return total_random_effect, tuple(mu)
    
    def _solver(self, marginal_residual: np.ndarray, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Solve the system of equations to obtain precision-weighted residuals.

        Parameters
        ----------
        marginal_residual : np.ndarray
            Marginal residuals.
        random_effects : tuple of RandomEffect
            Random effect objects for each grouping factor.
        residual : Residual
            Residual object.

        Returns
        -------
        prec_resid : np.ndarray
            Precision-weighted residuals.
        V_op : VLinearOperator
            Linear operator representing the marginal covariance.
        M_op : ResidualPreconditioner or None
            Preconditioner for the CG solver.
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
        Perform the E-step of the EM algorithm.

        Parameters
        ----------
        marginal_residual : np.ndarray
            Marginal residuals.
        random_effects : tuple of RandomEffect
            Random effect objects for each grouping factor.
        residual : Residual
            Residual object.

        Returns
        -------
        total_random_effect : np.ndarray or None
            Sum of all random effects, or None if converged.
        mu : tuple of np.ndarray or None
            Random effect coefficients, or None if converged.
        V_op : VLinearOperator or None
            Linear operator for marginal covariance, or None if converged.
        M_op : ResidualPreconditioner or None
            Preconditioner for CG solver, or None if converged.
        """
        prec_resid, V_op, M_op = self._solver(marginal_residual, random_effects, residual)
        current_log_lh = self._compute_log_likelihood(marginal_residual, prec_resid, V_op)
        self.log_likelihood.append(current_log_lh)
        if len(self.log_likelihood) >= 2:
            self._check_convergence(random_effects, residual)

        if self._is_converged:
            return None, None, None, None
        
        total_random_effect, mu = self._aggregate_random_effects(prec_resid, random_effects)

        return total_random_effect, mu, V_op, M_op

    def _m_step(self, marginal_residual: np.ndarray, total_random_effect: np.ndarray, mu: tuple[np.ndarray],
                random_effects: tuple[RandomEffect, ...], residual: Residual, V_op: VLinearOperator, M_op: ResidualPreconditioner):
        """
        Perform the M-step of the EM algorithm.

        Parameters
        ----------
        marginal_residual : np.ndarray
            Marginal residuals.
        total_random_effect : np.ndarray
            Sum of all random effects.
        mu : tuple of np.ndarray
            Random effect coefficients for each grouping factor.
        random_effects : tuple of RandomEffect
            Random effect objects for each grouping factor.
        residual : Residual
            Residual object.
        V_op : VLinearOperator
            Linear operator for marginal covariance.
        M_op : ResidualPreconditioner
            Preconditioner for CG solver.

        Returns
        -------
        self : MixedEffectRegressor
            Returns self for method chaining.
        """
        eps = marginal_residual - total_random_effect
        T_sum = np.zeros((self.m, self.m))
        new_tau = []
        for k, re in enumerate(random_effects):
            T_k, W_k = compute_cov_correction(k, V_op, M_op, self.correction_method, self.n_jobs, self.backend)
            T_sum += T_k
            new_tau.append(re._compute_cov(mu[k], W_k))

        residual.cov = residual._compute_cov(eps, T_sum)
        for k, re in enumerate(random_effects):
            re.cov = new_tau[k]

        return self

    def _check_convergence(self, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Check if the model parameters have converged.

        Parameters
        ----------
        random_effects : tuple of RandomEffect
            Random effect objects for each grouping factor.
        residual : Residual
            Residual object.

        Returns
        -------
        self : MixedEffectRegressor
            Returns self for method chaining.
        """
        change = np.abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2])
        self._is_converged = change <= self.tol

        current_log_lh = self.log_likelihood[-1]
        if current_log_lh > self._best_log_likelihood:
            self._best_log_likelihood = current_log_lh
            self._no_improvement_count = 0

            self._best_re_covs = [re.cov.copy() for re in random_effects]
            self._best_resid_cov = residual.cov.copy()
            self._best_fe_model = pickle.loads(pickle.dumps(self.fe_model))
        else:
            self._no_improvement_count += 1
        
        if self._no_improvement_count >= self.patience:
            for k, re in enumerate(random_effects):
                re.cov = self._best_re_covs[k]
            residual.cov = self._best_resid_cov
            self.fe_model = self._best_fe_model
            self._is_converged = True

        return self

    def _run_em_iteration(self, X: np.ndarray, y: np.ndarray, marginal_residual: np.ndarray, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Run a single EM iteration for fitting the model.

        Parameters
        ----------
        X : np.ndarray
            Fixed effect covariates of shape (n_samples, n_features).
        y : np.ndarray
            Response variables of shape (n_samples, n_outputs).
        marginal_residual : np.ndarray
            Marginal residuals.
        random_effects : tuple of RandomEffect
            Random effect objects for each grouping factor.
        residual : Residual
            Residual object.

        Returns
        -------
        marginal_residual : np.ndarray
            Updated marginal residuals for next iteration.
        """
        total_random_effect, mu, V_op, M_op = self._e_step(marginal_residual, random_effects, residual)
        if self._is_converged:
            return marginal_residual

        marginal_residual = self._compute_marginal_residual(X, y, total_random_effect.reshape((self.m, self.n)).T)

        self._m_step(marginal_residual, total_random_effect, mu, random_effects, residual, V_op, M_op)

        return marginal_residual

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | tuple[list[int]] = None):
        """
        Fit the mixed effects model using the EM algorithm.

        Parameters
        ----------
        X : np.ndarray
            Fixed effect covariates of shape (n_samples, n_features).
        y : np.ndarray
            Response variables of shape (n_samples, n_outputs).
        groups : np.ndarray
            Grouping factors of shape (n_samples, n_groups).
        random_slopes : None or tuple of list of int, optional
            Random slope column indices for each grouping factor.

        Returns
        -------
        MixedEffectResults
            Contains fitted model and results.
        """
        return self._fit(X=X, y=y, groups=groups, random_slopes=random_slopes)
    
    def partial_fit(self, marginal_residual: np.ndarray, random_effects: tuple[RandomEffect, ...], residual: Residual):
        """
        Continue fitting the model using previously fitted model parameters.

        Parameters
        ----------
        marginal_residual : np.ndarray
            Marginal residuals from previous fit.
        random_effects : tuple of RandomEffect
            Random effect objects from previous fit.
        residual : Residual
            Residual object from previous fit.

        Returns
        -------
        MixedEffectResults
            Contains fitted model and results.
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
        for _ in pbar:
            marginal_residual = self._run_em_iteration(X, y, marginal_residual, random_effects, residual)
            if self._is_converged:
                pbar.set_description(f"Model Converged | Early stopping after {self._no_improvement_count} iterations.")
                break
        # to avoid cicular import
        from .mixed_result import MixedEffectResults
        return MixedEffectResults(self, random_effects, residual)
