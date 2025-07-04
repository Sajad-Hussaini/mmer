import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.base import RegressorMixin, clone
from tqdm import tqdm
from . import linalg_op
from . import stats
from . import slq
from .random_effect import RandomEffect
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

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: dict):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        self.n, self.m = y.shape
        self.k = groups.shape[1]
        self.random_slopes = random_slopes if random_slopes is not None else {k: None for k in range(self.k)}

        rand_effects = {k: RandomEffect(self.n, self.m, k, slope_col) for k, slope_col in self.random_slopes.items()}
        for re in rand_effects.values():
            re.design_rand_effect(X, groups).design_stats().cross_product()

        self.resid_cov = np.eye(self.m)
        
        self.fe_models = [clone(self.fe_model) for _ in range(self.m)] if not isinstance(self.fe_model, list) else self.fe_model
        if len(self.fe_models) != self.m:
            raise ValueError(f"Expected {self.m} fixed effects models, got {len(self.fe_models)}")
        
        resid_mrg = self.compute_marginal_residual(X, y, 0.0)
        return resid_mrg, rand_effects
    
    def compute_marginal_residual(self, X: np.ndarray, y: np.ndarray, rand_effects: np.ndarray):
        """
        Compute marginal residuals by fitting the fixed effects models to the adjusted response variables.
        returns:
            2d array (n, M)
        """
        y_adj = y - rand_effects
        for m, model in enumerate(self.fe_models):
            y_adj[:, m] = model.fit(X, y_adj[:, m]).predict(X)
        return y - y_adj

    def compute_resid_cov(self, eps: np.ndarray, random_effects: dict[int, RandomEffect], V_op: LinearOperator, M_op: LinearOperator):
        """
        Compute the residual covariance matrix.
        Uses symmetry of the covariance matrix to reduce computations.
        """
        cov = eps.T @ eps
        for re in random_effects.values():
            np.add(cov, re.compute_resid_cov_correction(V_op, M_op, self.n_jobs).resid_cov_correction, out=cov)

        self.resid_cov[...] = cov / self.n + 1e-6 * np.eye(self.m)
        return self

    def compute_rand_effect_cov(self, random_effects: dict[int, RandomEffect], V_op: LinearOperator, M_op: LinearOperator):
        """
        Compute the random effects covariance matrix.
        """
        for re in random_effects.values():
            re.compute_rand_effect_cov(V_op, M_op, self.n_jobs)
        return self
    
    def compute_log_likelihood(self, resid_mrg: np.ndarray, prec_resid: np.ndarray, V_op: LinearOperator):
        """
        Compute the log-likelihood of the marginal distribution of the residuals.
        aka the marginal log-likelihood
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

    def compute_mu(self, prec_resid: np.ndarray, random_effects: dict[int, RandomEffect]):
        """
        Compute the conditional mean of the random effects by leveraging the kronecker structure.
        """
        for re in random_effects.values():
            re.compute_mu(prec_resid)
    
    def aggregate_rand_effects(self, random_effects: dict[int, RandomEffect]):
        """
        Computes the sum of all random effect contributions in observation space Σ_{k=1}^K (I_M ⊗ Z_κ) μ_κ.
        returns:
            2d array (n, M)
        """
        total_re = np.zeros((self.n, self.m))
        for re in random_effects.values():
            np.add(total_re, re.map_mu(), out=total_re)
        return total_re

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: dict = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slopes: dict of {group_id: X_cols} for random slopes per group (optional).

        Returns:
            MERMResult: Contains fitted model parameters and results.
        """
        resid_mrg, rand_effects = self.prepare_data(X, y, groups, random_slopes)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting model...", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            rhs = resid_mrg.ravel(order='F')
            V_op = linalg_op.VLinearOperator(rand_effects, self.resid_cov)
            M_op = linalg_op.ResidualPreconditioner(self.resid_cov, self.n)
            # M_op = linalg_op.DiagonalPreconditioner(rand_effects, self.resid_cov)
            # M_op = None
            prec_resid, _ = cg(V_op, rhs, M=M_op)
            self.compute_mu(prec_resid, rand_effects)

            log_likelihood = self.compute_log_likelihood(rhs, prec_resid, V_op)
            self.log_likelihood.append(log_likelihood)
            if iter_ > 2 and abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2]) < self.tol:
                pbar.set_description("Model Converged")
                self._is_converged = True
                break

            total_re = self.aggregate_rand_effects(rand_effects)
            resid_mrg = self.compute_marginal_residual(X, y, total_re)
            eps = resid_mrg - total_re
            self.compute_resid_cov(eps, rand_effects, V_op, M_op)
            self.compute_rand_effect_cov(rand_effects, V_op, M_op)
        return MERMResult(self, rand_effects)
