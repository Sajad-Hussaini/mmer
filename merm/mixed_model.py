import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.base import RegressorMixin, clone
from tqdm import tqdm
from . import utils
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
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 50, tol: float = 1e-4):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihood = []
        self._is_converged = False

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: dict):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        self.n_obs, self.n_res = y.shape
        self.n_groups = groups.shape[1]
        self.random_slopes = random_slopes if random_slopes is not None else {k: None for k in range(self.n_groups)}

        rand_effects = {k: RandomEffect(self.n_obs, self.n_res, k, slope) for k, slope in self.random_slopes.items()}
        for re in rand_effects.values():
            re.design_matrix(X, groups).crossproduct()

        self.resid_cov = np.eye(self.n_res)
        
        self.fe_models = [clone(self.fe_model) for _ in range(self.n_res)] if not isinstance(self.fe_model, list) else self.fe_model
        if len(self.fe_models) != self.n_res:
            raise ValueError(f"Expected {self.n_res} fixed effects models, got {len(self.fe_models)}")
        
        marg_resid = self.compute_marginal_residual(X, y, 0.0)
        return marg_resid, rand_effects
    
    def compute_marginal_residual(self, X: np.ndarray, y: np.ndarray, rand_effects: np.ndarray):
        """
        Compute marginal residuals by fitting the fixed effects models to the adjusted response variables.
        """
        y_adj = y - rand_effects
        for m, model in enumerate(self.fe_models):
            y_adj[:, m] = model.fit(X, y_adj[:, m]).predict(X)
        return y - y_adj

    def compute_resid_cov(self, eps: np.ndarray, random_effects: dict[int, RandomEffect], V_op: LinearOperator):
        """
        Compute the residual covariance matrix.
        Uses symmetry of the covariance matrix to reduce computations.
        """
        cov = eps.T @ eps
        for re in random_effects.values():
            np.add(cov, utils.resid_cov(re, V_op), out=cov)

        self.resid_cov = cov / self.n_obs + 1e-6 * np.eye(self.n_res)

    def compute_rand_effect_cov(self, random_effects: dict[int, RandomEffect], V_op: LinearOperator):
        """
        Compute the random effects covariance matrix.
        """
        for re in random_effects.values():
            re.cov = utils.rand_effect_cov(re, V_op)
    
    def compute_log_likelihood(self, marg_resid: np.ndarray, V_inv_eps: np.ndarray, V_op: LinearOperator):
        """
        Compute the log-likelihood of the marginal distribution of the residuals.
        aka the marginal log-likelihood
        """
        log_det_V = utils.slq_logdet(V_op)
        log_likelihood = -(self.n_res * self.n_obs * np.log(2 * np.pi) + log_det_V + marg_resid.T @ V_inv_eps) / 2
        return log_likelihood
    
    def compute_marginal_covariance(self, residual, random_effects, n_obs, n_level, Z_matrices):
        """
        Compute the marginal covariance matrix V and the random effects covariance matrices D.
        """
        V = sparse.kron(residual.cov, sparse.eye_array(n_obs, format='csr'), format='csr')
        for k in range(self.n_groups):
            D_k = sparse.kron(random_effects[k].cov, sparse.eye_array(random_effects[k].n_level, format='csr'), format='csr')
            Z_block_k = utils.block_diag_design_matrix(self.n_res, Z_matrices[k])
            V += Z_block_k @ D_k @ Z_block_k.T
        return V

    def compute_mu(self, V_inv_eps: np.ndarray, random_effects: dict[int, RandomEffect]):
        """
        Compute the conditional mean of the random effects by leveraging the kronecker structure.
        """
        for re in random_effects.values():
            utils.cond_mean(V_inv_eps, re)
    
    def aggregate_rand_effects(self, random_effects: dict[int, RandomEffect]):
        """
        Computes the sum of all random effect contributions in observation space.
        """
        total_re = np.zeros((self.n_obs, self.n_res))
        for re in random_effects.values():
            np.add(total_re, re.map_cond_mean(), out=total_re)
        return total_re

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: dict = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slopes: List of (n_samples, q_k) arrays for random slopes per group (optional).

        Returns:
            Self (fitted model).
        """
        marg_resid, rand_effects = self.prepare_data(X, y, groups, random_slopes)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting model...", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            rhs = marg_resid.T.ravel()
            V_op = utils.VLinearOperator(rand_effects, self.resid_cov, self.n_res, self.n_obs)
            V_inv_eps, _ = cg(V_op, rhs)
            self.compute_mu(V_inv_eps, rand_effects)

            log_likelihood = self.compute_log_likelihood(rhs, V_inv_eps, V_op)
            self.log_likelihood.append(log_likelihood)
            if iter_ > 2 and abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2]) < self.tol:
                pbar.set_description("Model Converged")
                self._is_converged = True
                break

            total_re = self.aggregate_rand_effects(rand_effects)
            marg_resid = self.compute_marginal_residual(X, y, total_re)
            eps = marg_resid - total_re
            self.compute_resid_cov(eps, rand_effects, V_op)
            self.compute_rand_effect_cov(rand_effects, V_op)

        return MERMResult(self, rand_effects)
