from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.base import RegressorMixin, clone
from tqdm import tqdm
from . import utils

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
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_indices: dict):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        self.num_obs, self.num_responses = y.shape
        self.num_groups = groups.shape[1]
        self.slope_indices = random_slope_indices

        z_matrices, self.num_random_effects, self.num_levels = utils.random_effect_design_matrices(X, groups, self.slope_indices)
        Z_blocks = utils.block_diag_design_matrices(z_matrices, self.num_responses)
        Z_crossprods = utils.crossprod_design_matrices(z_matrices)

        self.residuals_covariance = np.eye(self.num_responses)
        self.random_effects_covariance = {k: np.eye(self.num_responses * effect_k) for k, effect_k in self.num_random_effects.items()}
        
        self.fe_models = [clone(self.fe_model) for _ in range(self.num_responses)] if not isinstance(self.fe_model, list) else self.fe_model
        if len(self.fe_models) != self.num_responses:
            raise ValueError(f"Expected {self.num_responses} fixed effects models, got {len(self.fe_models)}")
        
        marginal_residuals = self.compute_marginal_residuals(X, y, 0.0)
        cached_data = TempData(marginal_residuals, Z_blocks, Z_crossprods)
        return cached_data
    
    def compute_marginal_residuals(self, X, y, effects_sum):
        """
        Compute the marginal residuals by fitting the fixed effects models to the adjusted response variables.
        """
        y_adj = y - effects_sum
        fX = np.zeros_like(y_adj)
        for m in range(self.num_responses):
            fX[:, m] = self.fe_models[m].fit(X, y_adj[:, m]).predict(X)
        marginal_residuals = y - fX
        return marginal_residuals

    def e_step(self, cached_data):
        """
        Perform the E-step of the EM algorithm to compute the conditional expectation and covariance of the random effects.
        """
        splu, D = self.splu_decomposition(self.num_obs, self.num_levels, cached_data.Z_blocks)
        V_inv_eps = splu.solve(cached_data.marginal_residuals.ravel(order='F'))
        log_likelihood = self.compute_log_likelihood(cached_data.marginal_residuals.ravel(order='F'), V_inv_eps, splu)
        mu = self.compute_mu(V_inv_eps, D, cached_data.Z_blocks)
        sigma = self.compute_sigma(D, splu, cached_data.Z_blocks)
        cached_data.mu = mu
        cached_data.sigma = sigma
        cached_data.log_likelihood = log_likelihood
        return cached_data
    
    def m_step(self, X, y, cached_data):
        """
        Perform the E-step of the EM algorithm to update the fixed effects functions, residual, and random effects covariance matrices.
        """
        effects_sum = self.sum_random_effects(cached_data.mu, cached_data.Z_blocks, self.num_obs)
        marginal_residuals = self.compute_marginal_residuals(X, y, effects_sum)
        eps = marginal_residuals - effects_sum
        self.compute_residuals_covariance(cached_data.sigma, eps, cached_data.Z_crossprods)
        self.compute_random_effects_covariance(cached_data.mu, cached_data.sigma)
        cached_data.marginal_residuals = marginal_residuals
        return cached_data
    
    def compute_residuals_covariance(self, sigma, eps, Z_crossprods):
        """
        the residual covariance matrix residuals_covariance.
        """
        S = eps.T @ eps
        T = np.zeros((self.num_responses, self.num_responses))
        for m1 in range(self.num_responses):
            for m2 in range(self.num_responses):
                trace_sum = 0.0
                for k in range(self.num_groups):
                    o_k, q_k = self.num_levels[k], self.num_random_effects[k]
                    idx1 = slice(m1 * q_k * o_k, (m1 + 1) * q_k * o_k)
                    idx2 = slice(m2 * q_k * o_k, (m2 + 1) * q_k * o_k)
                    sigma_k_block = sigma[k][idx1, idx2]
                    trace_sum += (Z_crossprods[k] @ sigma_k_block).trace()
                T[m1, m2] = trace_sum
        self.residuals_covariance = (S + T) / self.num_obs + 1e-6 * np.eye(self.num_responses)

    def compute_random_effects_covariance(self, mu, sigma):
        """
        Update the random effects covariance matrix random_effects_covariance.
        """
        for k in range(self.num_groups):
            o_k, q_k = self.num_levels[k], self.num_random_effects[k]
            mu_k = mu[k]
            sigma_k = sigma[k]
            sum_tau = np.zeros((self.num_responses * q_k, self.num_responses * q_k))
            for j in range(o_k):
                indices = []  # Indices for level j across all responses and effect types
                for m in range(self.num_responses):
                    for q in range(q_k):
                        idx = m * q_k * o_k + q * o_k + j
                        indices.append(idx)
                mu_k_j = mu_k[indices]
                sigma_k_block = sigma_k[np.ix_(indices, indices)]
                sum_tau += np.outer(mu_k_j, mu_k_j) + sigma_k_block
            self.random_effects_covariance[k] = sum_tau / o_k + 1e-6 * np.eye(self.num_responses * q_k)

    def compute_log_likelihood(self, marginal_residuals, V_inv_eps, splu):
        """
        Compute the log-likelihood of the marginal distribution of the residuals (the marginal log-likelihood)
        """
        log_det_V = np.sum(np.log(np.abs(splu.U.diagonal())))
        log_likelihood = -(self.num_responses * self.num_obs * np.log(2 * np.pi) + log_det_V + marginal_residuals.T @ V_inv_eps) / 2
        return log_likelihood
    
    def compute_marginal_covariance(self, n_obs, n_level, Z_blocks):
        """
        Compute the marginal covariance matrix V and the random effects covariance matrices D.
        """
        D = {k: sparse.kron(tau_k, sparse.eye_array(n_level[k], format='csr'), format='csr') for k, tau_k in self.random_effects_covariance.items()}
        V = sparse.kron(self.residuals_covariance, sparse.eye_array(n_obs, format='csr'), format='csr')
        for k, D_k in D.items():
            V += Z_blocks[k] @ D_k @ Z_blocks[k].T
        return V, D

    def splu_decomposition(self, n_obs, n_level, Z_blocks):
        """
        Compute the sparse LU decomposition of the marginal covariance matrix V.
        """
        V, D = self.compute_marginal_covariance(n_obs, n_level, Z_blocks)
        return sparse.linalg.splu(V.tocsc()), D

    @staticmethod
    def compute_mu(V_inv_eps, D, Z_blocks):
        """
        Compute the conditional mean of the random effects.
        """
        return {k: D_k @ Z_blocks[k].T @ V_inv_eps for k, D_k in D.items()}

    @staticmethod
    def compute_sigma(D, splu, Z_blocks):
        """
        Compute the conditional covariance of the random effects.
        """
        sigma = {}
        for k, D_k in D.items():
            Im_Z_k = Z_blocks[k]
            Im_Z_D = Im_Z_k @ D_k
            V_inv_Im_Z_D = sparse.csr_array(splu.solve(Im_Z_D.toarray()))
            sigma[k] = D_k - D_k @ Im_Z_k.T @ V_inv_Im_Z_D
        return sigma

    def sum_random_effects(self, mu, Z_blocks, n_obs):
        """
        Compute the sum of random effects contributions for all groups.
        """
        effects_sum = np.zeros((n_obs, self.num_responses))
        for k, mu_k in mu.items():
            effects_sum += (Z_blocks[k] @ mu_k).reshape((n_obs, self.num_responses), order='F')
        return effects_sum

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_indices: dict = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slope_indices: List of (n_samples, q_k) arrays for random slopes per group (optional).

        Returns:
            Self (fitted model).
        """
        cached_data = self._prepare_data(X, y, groups, random_slope_indices)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Model Fitting", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            cached_data = self.e_step(cached_data)
            self.log_likelihood.append(cached_data.log_likelihood)
            if iter_ > 2 and abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2]) < self.tol:
                pbar.set_description("Model Converged")
                self._is_converged = True
                break
            cached_data = self.m_step(X, y, cached_data)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses using the fitted fixed effects models.
        
        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
        
        Returns:
            (n_samples, M) array of predicted responses.
        """
        if not self.fe_models:
            raise ValueError("Model must be fitted before prediction.")
        n = X.shape[0]
        fX = np.zeros((n, self.num_responses))
        for m in range(self.num_responses):
            fX[:, m] = self.fe_models[m].predict(X)
        return fX
    
    def sample(self, X: np.ndarray) -> np.ndarray:
        """
        Sample responses from the predictive multivariate distribution.
        
        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
        
        Returns:
            (n_samples, M) array of sampled responses.
        """
        fX = self.predict(X)
        n = X.shape[0]
        y_sampled = np.zeros_like(fX)
        groups = np.zeros((n, self.num_groups), dtype=int)
        for i in range(n):
            z_matrices, _, n_level = utils.random_effect_design_matrices(X[i:i+1], groups[i:i+1], self.slope_indices)
            Z_blocks = utils.block_diag_design_matrices(z_matrices, self.num_responses)
            V_i, _ = self.compute_marginal_covariance(1, n_level, Z_blocks)
            y_sampled[i] = np.random.multivariate_normal(fX[i], V_i)
        return y_sampled
    
    def compute_random_effects_and_residuals(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Compute residuals (num_obs x num_obs) and random effects (num_responses x num_random_effects x num_levels).
        """
        n_obs, _ = y.shape
        z_matrices, _, n_level = utils.random_effect_design_matrices(X, groups, self.slope_indices)
        Z_blocks = utils.block_diag_design_matrices(z_matrices, self.num_responses)
        splu, D = self.splu_decomposition(n_obs, n_level, Z_blocks)

        marginal_residuals = y - self.predict(X)
        V_inv_eps = splu.solve(marginal_residuals.ravel(order='F'))
        mu = self.compute_mu(V_inv_eps, D, Z_blocks)

        effects_sum = self.sum_random_effects(mu, Z_blocks, n_obs)
        eps = marginal_residuals - effects_sum
        return mu, eps

    def summary(self):
        """
        Display a summary of the fitted multivariate mixed effects model.
        """
        if not self.fe_models:
            raise ValueError("Model must be fitted before calling summary.")

        # Print summary statistics
        indent0 = ""
        indent1 = "   "
        indent2 = "       "

        print("\n" + indent0 + "Multivariate Mixed Effects Model Summary")
        print("=" * 50)
        print(indent1 + f"FE Model: {type(self.fe_models[0]).__name__}")
        print(indent1 + f"Iterations: {len(self.log_likelihood)}")
        print(indent1 + f"Converged: {self._is_converged}")
        print(indent1 + f"Log-Likelihood: {self.log_likelihood[-1]:.2f}")
        print(indent1 + f"No. Observations: {self.num_obs}")
        print(indent1 + f"No. Response Variables: {self.num_responses}")
        print(indent1 + f"No. Grouping Variables: {self.num_groups}")
        print("-" * 50)
        print(indent1 + f"Residual (Unexplained) Variances")
        print(indent2 + "{:<10} {:>10}".format("Response", "Variance"))
        for m in range(self.num_responses):
            print(indent2 + "{:<10} {:>10.4f}".format(m + 1, self.residuals_covariance[m, m]))
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        print(indent2 + "{:<8} {:<10} {:<15} {:>10}".format("Group", "Response", "Random Effect", "Variance"))
        for k in range(self.num_groups):
            for i in range(self.num_responses):
                for j in range(self.num_random_effects[k]):
                    idx = i * self.num_random_effects[k] + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    var = self.random_effects_covariance[k][idx, idx]
                    print(indent2 + "{:<8} {:<10} {:<15} {:>10.4f}".format(k + 1, i + 1, effect_name, var))
        print("\n")

@dataclass
class TempData:
    """
    Temporary data structure to hold intermediate results during model fitting.
    """
    marginal_residuals: np.ndarray
    Z_blocks: dict
    Z_crossprods: dict
