import numpy as np
from . import utils

class MERMResult:
    """
    Result class for the Multivariate Mixed Effects Regression Model (MERM).
    """
    def __init__(self, fitted_model, random_effects):
        self.fe_models = fitted_model.fe_models

        self.n_res = fitted_model.n_res
        self.n_obs = fitted_model.n_obs
        self.n_groups = fitted_model.n_groups
        self.random_slopes = fitted_model.random_slopes
        
        self.rand_effects = random_effects
        self.resid_cov = fitted_model.resid_cov
        self.log_likelihood = fitted_model.log_likelihood
        self._is_converged = fitted_model._is_converged
    
    @property
    def rand_effect_cov(self):
        """Lazy property to get covariance matrices when needed."""
        return {k: re.cov for k, re in self.rand_effects.items()}
    
    @property
    def n_effect(self):
        """Lazy property to get number of effects when needed."""
        return {k: re.n_effect for k, re in self.rand_effects.items()}
    
    @property
    def n_level(self):
        """Lazy property to get number of levels when needed."""
        return {k: re.n_level for k, re in self.rand_effects.items()}

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
        
        fX = np.empty((X.shape[0], self.n_res))
        for i, model in enumerate(self.fe_models):
            fX[:, i] = model.predict(X)
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
        groups = np.zeros((n, self.n_groups), dtype=int)
        for i in range(n):
            Z_matrices, _, n_level = utils.random_effect_design_matrices(X[i:i+1], groups[i:i+1], self.slope_indices)
            Z_blocks = utils.block_diag_design_matrices(Z_matrices, self.n_res)
            V_i, _ = self.compute_marginal_covariance(1, n_level, Z_blocks)
            y_sampled[i] = np.random.multivariate_normal(fX[i], V_i)
        return y_sampled
    
    def compute_random_effects_and_residuals(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Compute residuals (n_obs x n_obs) and random effects (n_res x n_effect x num_levels).
        """
        self.n_obs, _ = y.shape
        Z_matrices, _, n_level = utils.random_effect_design_matrices(X, groups, self.slope_indices)
        Z_blocks = utils.block_diag_design_matrices(Z_matrices, self.n_res)
        splu, D = self.splu_decomposition(self.n_obs, n_level, Z_blocks)

        marg_resid = y - self.predict(X)
        V_inv_eps = splu.solve(marg_resid.ravel(order='F'))
        mu = self.compute_mu(V_inv_eps, D, Z_blocks)

        total_re = self.aggregate_rand_effects(mu, Z_matrices)
        eps = marg_resid - total_re
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
        print(indent1 + f"No. Observations: {self.n_obs}")
        print(indent1 + f"No. Response Variables: {self.n_res}")
        print(indent1 + f"No. Grouping Variables: {self.n_groups}")
        print("-" * 50)
        print(indent1 + f"Residual (Unexplained) Variances")
        print(indent2 + "{:<10} {:>10}".format("Response", "Variance"))
        for m in range(self.n_res):
            print(indent2 + "{:<10} {:>10.4f}".format(m + 1, self.resid_cov[m, m]))
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        print(indent2 + "{:<8} {:<10} {:<15} {:>10}".format("Group", "Response", "Random Effect", "Variance"))
        for k in range(self.n_groups):
            for i in range(self.n_res):
                for j in range(self.n_effect[k]):
                    idx = i * self.n_effect[k] + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    var = self.rand_effect_cov[k][idx, idx]
                    print(indent2 + "{:<8} {:<10} {:<15} {:>10.4f}".format(k + 1, i + 1, effect_name, var))
        print("\n")