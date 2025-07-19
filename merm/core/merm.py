import numpy as np
from scipy.sparse.linalg import cg
from sklearn.base import RegressorMixin
from tqdm import tqdm
from .operator import VLinearOperator, ResidualPreconditioner
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
        max_iter: Maximum number iterations (default: 20).
        tol: Log-likelihood convergence tolerance  (default: 1e-6).
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 20, tol: float = 1e-6,
                 slq_steps: int = 5, slq_probes: int = 5, n_jobs: int = 4, backend: str = 'threading'):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.slq_steps = slq_steps
        self.slq_probes = slq_probes
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
            re.design_rand_effect(X, groups).prepare_data()

        resid = Residual(self.n, self.m)
        
        resid_mrg = self.compute_marginal_residual(X, y, 0.0)
        return resid_mrg, rand_effects, resid
    
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

    def compute_log_likelihood(self, resid_mrg: np.ndarray, prec_resid: np.ndarray, V_op: VLinearOperator):
        """
        Compute the log-likelihood of the marginal distribution of the residuals.
        aka the marginal log-likelihood
            resid_mrg: marginal residuals y-fx
            prec_resid: precision-weighted residuals V⁻¹(y-fx)
        """
        log_det_V = slq.logdet(V_op, self.slq_steps, self.slq_probes, self.n_jobs, self.backend)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + resid_mrg.T @ prec_resid) / 2
        return log_likelihood

    def aggregate_rand_effects(self, random_effects: dict[int, RandomEffect], prec_resid: np.ndarray):
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

    def _e_step(self, random_effects: dict[int, RandomEffect], residual: Residual, resid_mrg: np.ndarray):
        """Performs the E-step of the EM algorithm."""
        old_tau = {k: re.cov.copy() for k, re in random_effects.items()}
        old_phi = residual.cov.copy()
        V_op = VLinearOperator(random_effects, residual)
        M_op = ResidualPreconditioner(residual)
        prec_resid, _ = cg(V_op, resid_mrg, M=M_op)
        total_re, mu = self.aggregate_rand_effects(random_effects, prec_resid)
        return total_re, mu, old_tau, old_phi, V_op, M_op

    def _m_step(self, random_effects: dict[int, RandomEffect], residual: Residual, resid_mrg: np.ndarray,
                total_re, mu, V_op, M_op):
        """Performs the M-step of the EM algorithm."""
        eps = np.subtract(resid_mrg, total_re, out=total_re)
        new_tau = {}
        T_sum = np.zeros((self.m, self.m))
        for k, re in random_effects.items():
            T_k, W_k = re.compute_cov_correction(V_op, M_op, self.n_jobs, self.backend)
            np.add(T_sum, T_k, out=T_sum)
            new_tau[k] = re.compute_cov(mu[k], W_k)

        residual.cov[...] = residual.compute_cov(eps, T_sum)
        for k, re in random_effects.items():
            re.cov[...] = new_tau[k]

    def _check_convergence(self, random_effects, residual, old_tau, old_phi):
        """Checks if the model parameters have converged."""
        phi_change = np.linalg.norm(residual.cov - old_phi) / np.linalg.norm(old_phi)
        tau_changes = [np.linalg.norm(re.cov - old_tau[k]) / np.linalg.norm(old_tau[k])
                       for k, re in random_effects.items()]
        max_param_change = max([phi_change] + tau_changes)
        return max_param_change, max_param_change < self.tol

    def _run_em_iteration(self, X, y, rand_effects, resid, resid_mrg):
        """ Run the EM algorithm for fitting the model. """
        # --- E-Step ---
        total_re, mu, old_tau, old_phi, V_op, M_op = self._e_step(rand_effects, resid, resid_mrg)

        # --- Update marginal residual for M-step ---
        resid_mrg = self.compute_marginal_residual(X, y, total_re.reshape((self.m, self.n)).T)

        # --- M-Step ---
        self._m_step(rand_effects, resid, resid_mrg, total_re, mu, V_op, M_op)

        return resid_mrg, old_tau, old_phi
    
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
        resid_mrg, rand_effects, resid = self.prepare_data(X, y, groups, random_slopes)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            resid_mrg, old_tau, old_phi = self._run_em_iteration(X, y, rand_effects, resid, resid_mrg)
            # --- Convergence Check ---
            if iter_ > 2:
                change, converged = self._check_convergence(rand_effects, resid, old_tau, old_phi)
                pbar.set_postfix_str(f"Max Param Change: {change:.2e}")
                if converged:
                    pbar.set_description("Model Converged")
                    self._is_converged = True
                    break
        
        V_op = VLinearOperator(rand_effects, resid)
        M_op = ResidualPreconditioner(resid)
        prec_resid, _ = cg(V_op, resid_mrg, M=M_op)
        final_logL = self.compute_log_likelihood(resid_mrg, prec_resid, V_op)
        self.log_likelihood.append(final_logL)
        return MERMResult(self, rand_effects, resid)
