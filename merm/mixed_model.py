import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from sklearn.base import RegressorMixin, clone
from tqdm import tqdm
from .linop import VLinearOperator, ResidualPreconditioner, DiagonalPreconditioner
from . import slq
from .random_effect import RandomEffect
from .residual import Residual
from .mixed_model_result import MERMResult

class MERM:
    """
    Multivariate Mixed Effects Regression Model.
    It supports multiple responses, any fixed effects model, multiple random effects, and multiple grouping factors.
    Parameters:
        fixed_effects_model: A scikit-learn regressor or list of regressors for fixed effects.
        max_iter: Maximum number iterations (default: 50).
        tol: Log-likelihood convergence tolerance  (default: 1e-4).
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 50, tol: float = 1e-4, n_jobs: int = -2):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihood = []
        self._is_converged = False
        self.n_jobs = n_jobs

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
        
        self.fe_models = [clone(self.fe_model) for _ in range(self.m)] if not isinstance(self.fe_model, list) else self.fe_model
        if len(self.fe_models) != self.m:
            raise ValueError(f"Expected {self.m} fixed effects models, got {len(self.fe_models)}")
        
        resid_mrg = self.compute_marginal_residual(X, y, 0.0)
        return resid_mrg, rand_effects, resid
    
    def compute_marginal_residual(self, X: np.ndarray, y: np.ndarray, total_rand_effect: np.ndarray):
        """
        Compute marginal residuals by fitting the fixed effects models to the adjusted response variables.
        returns:
            2d array (n, M)
        """
        temp = y - total_rand_effect
        for m, model in enumerate(self.fe_models):
            temp[:, m] = model.fit(X, temp[:, m]).predict(X)
        np.subtract(y, temp, out=temp)
        return temp

    def compute_log_likelihood(self, resid_mrg: np.ndarray, prec_resid: np.ndarray, V_op: VLinearOperator):
        """
        Compute the log-likelihood of the marginal distribution of the residuals.
        aka the marginal log-likelihood
            resid_mrg: marginal residuals y-fx
            prec_resid: precision-weighted residuals V⁻¹(y-fx)
        """
        log_det_V = slq.logdet(V_op, n_jobs=self.n_jobs)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + resid_mrg.T @ prec_resid) / 2
        return log_likelihood
    
    def compute_marginal_covariance(self, random_effects, n):
        """
        Compute the marginal covariance matrix V
        """
        V = sparse.kron(self.resid_cov, sparse.eye_array(n, format='csr'), format='csr')
        for re in random_effects.values():
            D = sparse.kron(re.cov, sparse.eye_array(re.n_level, format='csr'), format='csr')
            Z_full = sparse.kron(sparse.eye_array(re.m, format='csr'), re.Z, format='csr')
            V += Z_full @ D @ Z_full.T
        return V
    
    def aggregate_rand_effects(self, random_effects: dict[int, RandomEffect]):
        """
        Computes sum of all random effects in observation space.
            Σₖ(Iₘ ⊗ Zₖ)μₖ
        returns:
            2d array (n, M)
        """
        total_re = np.zeros((self.n, self.m))
        for re in random_effects.values():
            np.add(total_re, re.map_mu(), out=total_re)
        return total_re

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
            rhs = resid_mrg.ravel(order='F')
            V_op = VLinearOperator(rand_effects, resid)
            # M_op = ResidualPreconditioner(resid)
            M_op = DiagonalPreconditioner(rand_effects, resid)
            prec_resid, _ = cg(V_op, rhs, M=M_op)

            for re in rand_effects.values():
                re.compute_mu(prec_resid)

            log_likelihood = self.compute_log_likelihood(rhs, prec_resid, V_op)
            self.log_likelihood.append(log_likelihood)
            if iter_ > 2 and abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2]) < self.tol:
                pbar.set_description("Model Converged")
                self._is_converged = True
                break

            total_re = self.aggregate_rand_effects(rand_effects)
            resid_mrg = self.compute_marginal_residual(X, y, total_re)

            resid.compute_eps(resid_mrg, total_re)
            new_phi = resid.compute_cov(rand_effects, V_op, M_op, self.n_jobs)
            tau_dict = {}
            for k, re in rand_effects.items():
                tau_dict[k] = re.compute_cov(V_op, M_op, self.n_jobs)
            # Safely update the covariance matrices
            resid.cov = new_phi
            for k, re in rand_effects.items():
                re.cov = tau_dict[k]
        return MERMResult(self, rand_effects, resid)
