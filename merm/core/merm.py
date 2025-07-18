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
            2d array (n, M)
        """
        y_adj = y - total_rand_effect
        if self.m == 1:
            fx = self.fe_model.fit(X, y_adj.ravel()).predict(X)[:, None]
        else:
            fx = self.fe_model.fit(X, y_adj).predict(X)
        return y - fx

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
            2d array (n, M)
        """
        total_re = np.zeros((self.n, self.m))
        mu = {}
        for k, re in random_effects.items():
            mu[k] = re.compute_mu(prec_resid)
            np.add(total_re, re.map_mu(mu[k]), out=total_re)
        return total_re, mu

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
        W = {}
        T = {}
        new_tau = {}
        old_tau = {}
        for iter_ in pbar:
            old_phi = resid.cov.copy()
            old_tau.clear()
            for k, re in rand_effects.items():
                old_tau[k] = re.cov.copy()

            rhs = resid_mrg.ravel(order='F')
            V_op = VLinearOperator(rand_effects, resid)
            M_op = ResidualPreconditioner(resid)
            prec_resid, _ = cg(V_op, rhs, M=M_op)

            total_re, mu = self.aggregate_rand_effects(rand_effects, prec_resid)
            resid_mrg = self.compute_marginal_residual(X, y, total_re)

            eps = resid_mrg - total_re
            W.clear()
            T.clear()
            new_tau.clear()
            for k, re in rand_effects.items():
                T[k], W[k] = re.compute_cov_correction(V_op, M_op, self.n_jobs, self.backend)
                new_tau[k] = re.compute_cov(mu[k], W[k])

            resid.cov[...] = resid.compute_cov(eps, T)
            for k, re in rand_effects.items():
                re.cov[...] = new_tau[k]
            
            if iter_ > 2:
                phi_change = np.linalg.norm(resid.cov - old_phi) / np.linalg.norm(old_phi)
                tau_changes = [np.linalg.norm(re.cov - old_tau[k]) / np.linalg.norm(old_tau[k]) for k, re in rand_effects.items()]
                max_param_change = max([phi_change] + tau_changes)
                pbar.set_postfix_str(f"Max Param Change: {max_param_change:.2e}")

                if max_param_change < self.tol:
                    pbar.set_description("Model Converged")
                    self._is_converged = True
                    break
        
        V_op = VLinearOperator(rand_effects, resid)
        M_op = ResidualPreconditioner(resid)
        rhs = resid_mrg.ravel(order='F')
        prec_resid, _ = cg(V_op, rhs, M=M_op)
        final_logL = self.compute_log_likelihood(rhs, prec_resid, V_op)
        self.log_likelihood.append(final_logL)
        return MERMResult(self, rand_effects, resid)
