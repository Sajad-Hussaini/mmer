import numpy as np
from scipy.sparse.linalg import cg
from sklearn.base import RegressorMixin
from tqdm import tqdm
from .operator import VLinearOperator, ResidualPreconditioner, compute_cov_correction
from ..lanczos_algorithm import slq
from .random_effect import RandomEffect
from .residual import Residual
from .merm_result import MERMResult

class MERM:
    """
    Multivariate Mixed Effects Regression Model.
    It supports multiple responses, any fixed effects model, multiple random effects, and multiple grouping factors.
    Parameters:
        fixed_effects_model: A scikit-learn regressor that supports multi-output regression.
        max_iter: Maximum number of EM iterations.
        tol: Log-likelihood convergence tolerance.
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 20, tol: float = 1e-5,
                 slq_steps: int = 25, slq_probes: int = 25, V_conditioner: bool = False, correction_method: str = 'bste',
                 n_jobs: int = 1, backend: str = 'loky'):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.slq_steps = slq_steps
        self.slq_probes = slq_probes
        self.V_conditioner = V_conditioner
        self.correction_method = correction_method
        existing_method = ['ste', 'bste', 'detr']
        if self.correction_method not in existing_method:
            raise ValueError(f"Unknown correction method: '{self.correction_method}'. Available methods are {existing_method}.")

        self.log_likelihood = []
        self._is_converged = False
        self.n_jobs = n_jobs
        self.backend = backend

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | dict[int, list[int]]):
        """
        Prepare initial parameters, instances of random effects and residuals.
        """
        self.n, self.m = y.shape
        self.k = groups.shape[1]
        self.random_slopes = random_slopes if random_slopes is not None else {k: None for k in range(self.k)}

        rand_effects = {k: RandomEffect(self.n, self.m, k, slope_col) for k, slope_col in self.random_slopes.items()}
        for re in rand_effects.values():
            re.design_rand_effect(X, groups)
        resid = Residual(self.n, self.m)
        resid_marginal = self.compute_marginal_residual(X, y, 0.0)
        return resid_marginal, rand_effects, resid
    
    def compute_marginal_residual(self, X: np.ndarray, y: np.ndarray, total_rand_effect: np.ndarray):
        """
        Compute marginal residuals by fitting the fixed effects models to the adjusted response variables.
        returns:
            1d array (mn,)
        """
        y_adj = y - total_rand_effect
        if self.m == 1:
            fx = self.fe_model.fit(X, y_adj.ravel()).predict(X)[:, None]
        else:
            fx = self.fe_model.fit(X, y_adj).predict(X)
        return (y - fx).T.ravel()

    def compute_log_likelihood(self, resid_marginal: np.ndarray, prec_resid: np.ndarray, V_op: VLinearOperator):
        """
        Compute the log-likelihood of the marginal distribution of the residuals.
        aka the marginal log-likelihood
            resid_marginal: marginal residuals y-fx
            prec_resid: precision-weighted residuals V⁻¹(y-fx)
        """
        log_det_V = slq.logdet(V_op, self.slq_steps, self.slq_probes, self.n_jobs, self.backend)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + resid_marginal.T @ prec_resid) / 2
        return log_likelihood

    def aggregate_rand_effects(self, prec_resid: np.ndarray, random_effects: dict[int, RandomEffect]):
        """
        Computes sum of all random effects in observation space.
            Σₖ(Iₘ ⊗ Zₖ)μₖ
        returns:
            1d array (mn,) and dict mu
        """
        total_re = np.zeros(self.m * self.n)
        mu = {}
        for k, re in random_effects.items():
            mu[k] = re.compute_mu(prec_resid)
            np.add(total_re, re.map_mu(mu[k]), out=total_re)
        return total_re, mu

    def _e_step(self, resid_marginal: np.ndarray, random_effects: dict[int, RandomEffect], residual: Residual):
        """Performs the E-step of the EM algorithm."""
        V_op = VLinearOperator(random_effects, residual)
        M_op = ResidualPreconditioner(residual) if self.V_conditioner else None
        prec_resid, _ = cg(A=V_op, b=resid_marginal, rtol=1e-5, atol=1e-8, maxiter=100, M=M_op)
        final_logL = self.compute_log_likelihood(resid_marginal, prec_resid, V_op)
        self.log_likelihood.append(final_logL)
        total_re, mu = self.aggregate_rand_effects(prec_resid, random_effects)
        return total_re, mu, V_op, M_op

    def _m_step(self, resid_marginal: np.ndarray, total_re: np.ndarray, mu: dict[int, np.ndarray],
                random_effects: dict[int, RandomEffect], residual: Residual, V_op: VLinearOperator, M_op: ResidualPreconditioner):
        """Performs the M-step of the EM algorithm."""
        eps = np.subtract(resid_marginal, total_re, out=total_re)
        new_tau = {}
        T_sum = np.zeros((self.m, self.m))
        for k, re in random_effects.items():
            T_k, W_k = compute_cov_correction(k, V_op, M_op, self.correction_method, self.n_jobs, self.backend)
            np.add(T_sum, T_k, out=T_sum)
            new_tau[k] = re.compute_cov(mu[k], W_k)

        residual.cov[...] = residual.compute_cov(eps, T_sum)
        for k, re in random_effects.items():
            re.cov[...] = new_tau[k]
        return self

    def _check_convergence(self):
        """Checks if the model parameters have converged."""
        tol = np.abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2])
        return tol < self.tol

    def _run_em_iteration(self, X: np.ndarray, y: np.ndarray, resid_marginal: np.ndarray, rand_effects: dict[int, RandomEffect], resid: Residual):
        """ Run the EM algorithm for fitting the model. """
        # --- E-Step ---
        total_re, mu, V_op, M_op = self._e_step(resid_marginal, rand_effects, resid)

        # --- Update marginal residual for M-step ---
        resid_marginal = self.compute_marginal_residual(X, y, total_re.reshape((self.m, self.n)).T)

        # --- M-Step ---
        self._m_step(resid_marginal, total_re, mu, rand_effects, resid, V_op, M_op)

        return resid_marginal
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | dict[int, list[int]] = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slopes: dict[int, list[int]] dictionary mapping group indices to lists of random slope indices (optional).

        Returns:
            MERMResult: Contains fitted model and results.
        """
        resid_marginal, rand_effects, resid = self.prepare_data(X, y, groups, random_slopes)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}", dynamic_ncols=True)
        for iter_ in pbar:
            resid_marginal = self._run_em_iteration(X, y, resid_marginal, rand_effects, resid)
            if iter_ > 2:
                converged = self._check_convergence()
                if converged:
                    pbar.set_description("Model Converged")
                    self._is_converged = True
                    break
        return MERMResult(self, rand_effects, resid)
