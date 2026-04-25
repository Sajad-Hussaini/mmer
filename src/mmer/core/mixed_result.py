import numpy as np
from .mixed_effect import MixedEffectRegressor
from .inference import compute_random_effects_posterior
from .terms import RealizedRandomEffect, RealizedResidual


class MixedEffectResults:
    """
    Result container for a fitted MMER model.

    Provides access to the fitted model state (coefficients, covariances) and 
    inference methods.

    Attributes
    ----------
    model : MixedEffectRegressor
        The source model instance.
    fe_model : RegressorMixin
        Fitted fixed effects model.
    m : int
        Number of output responses.
    k : int
        Number of grouping factors.
    random_effect_terms : list of RandomEffectTerm
        Learned random effect states.
    residual_term : ResidualTerm
        Learned residual state.
    log_likelihood : list
        History of log-likelihood values during training.
    """
    def __init__(self, model: MixedEffectRegressor):
        # Expose convenient attributes directly, decoupling from regressor model state
        self.fe_model = model.fe_model
        self.m = model.m
        self.k = model.k
        self.random_effect_terms = model.random_effect_terms
        self.residual_term = model.residual_term
        self.log_likelihood = model.convergence_monitor.log_likelihood
        self.is_converged = model.convergence_monitor.is_converged
        self.is_early_stopped = model.convergence_monitor.is_early_stopped
        self.best_log_likelihood = model.convergence_monitor._best_log_likelihood
        self.preconditioner = model.preconditioner
        self.cg_maxiter = model.cg_maxiter

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using fixed effects component only.
        
        Makes predictions using only the learned fixed effects model, ignoring
        random effects. For predictions that include random effects, use
        compute_random_effects() and add the total effect to predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Covariates for prediction, shape (n, p).

        Returns
        -------
        predictions : np.ndarray
            Predicted values from fixed effects only, shape (n, m).
        
        Notes
        -----
        Current implementation does not support random effects in prediction.
        To obtain predictions including random effects:
        
        1. Call compute_random_effects() to get random effect estimates
        2. Add total_effect to fixed effect predictions
        
        Examples
        --------
        >>> # Predict with fixed effects only
        >>> y_pred = results.predict(X_new)
        
        >>> # Predict with both fixed and random effects
        >>> y_fixed = results.predict(X_new)
        >>> _, total_re, _ = results.compute_random_effects(X_new, y_new, groups_new)
        >>> y_pred_full = y_fixed + total_re
        """
        return self.fe_model.predict(X)
    
    def compute_random_effects(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Estimate posterior random effects for new or existing data.

        Parameters
        ----------
        X : np.ndarray
            Features, shape (n, p).
        y : np.ndarray
            Targets, shape (n, m).
        groups : np.ndarray
            Grouping factors, shape (n, k).

        Returns
        -------
        residuals : np.ndarray
            Estimated residuals after accounting for fixed and random effects.
        total_effect : np.ndarray
            Total estimated random effects.
        mu : tuple of np.ndarray
            Estimated random effects for each grouping factor.
        """
        if self.random_effect_terms is None:
            raise RuntimeError("Model is not fitted.")
            
        n = X.shape[0]
        realized_effects = tuple(RealizedRandomEffect(term, X, groups) for term in self.random_effect_terms)
        realized_residual = RealizedResidual(self.residual_term, n)
        
        # Predict Fixed Effects
        fx = self.fe_model.predict(X)
        fx = fx if self.m != 1 else fx[:, None]
        
        return compute_random_effects_posterior(realized_effects, realized_residual, y, fx,
                                                self.preconditioner, self.cg_maxiter)

    @property
    def residual_covariance(self) -> np.ndarray:
        """Get the estimated residual covariance matrix."""
        return self.residual_term.cov
        
    @property
    def random_effects_covariances(self) -> tuple[np.ndarray]:
        """Get the estimated covariance matrices for each random effect grouping factor."""
        return [term.cov for term in self.random_effect_terms]

    @staticmethod
    def cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """
        Convert a covariance matrix to a correlation matrix.
        Protects against zero variances for numerical stability.
        """
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-12  # avoid division by zero
        return cov / np.outer(std, std)

    @property
    def residual_correlation(self) -> np.ndarray:
        """Get the estimated residual correlation matrix."""
        return self.cov_to_corr(self.residual_covariance)
        
    @property
    def random_effects_correlations(self) -> tuple[np.ndarray]:
        """Get the estimated correlation matrices for each random effect grouping factor."""
        return [self.cov_to_corr(cov) for cov in self.random_effects_covariances]

    def get_marginal_correlation(self, slope_covariates: tuple[np.ndarray | None] | None = None) -> np.ndarray:
        """
        Compute the total marginal correlation matrix (m x m) for a single observation profile.
        
        Parameters
        ----------
        slope_covariates : list of np.ndarray or None, optional
            List of 1D arrays containing random slope covariates for each grouping factor.
            Omit the intercept (1.0) as it is automatically included.
            
        Returns
        -------
        corr : np.ndarray
            Total marginal correlation matrix.
        """
        return self.cov_to_corr(self.get_marginal_covariance(slope_covariates))

    def get_marginal_covariance(self, slope_covariates: tuple[np.ndarray | None] | None = None) -> np.ndarray:
        """
        Compute the total marginal covariance matrix (m x m) for a single observation profile.
        
        Parameters
        ----------
        slope_covariates : list of np.ndarray or None, optional
            List of 1D arrays containing random slope covariates for each grouping factor.
            Omit the intercept (1.0) as it is automatically included. 
            If a group has only a random intercept, pass None or an empty array for that group.
            If entirely None, assumes random intercepts only for all groups.
            
        Returns
        -------
        cov : np.ndarray
            Total marginal covariance matrix combining residuals and random effects.
        """
        cov = self.residual_covariance.copy()
        if slope_covariates is None:
            slope_covariates = [None] * self.k
            
        for k, term in enumerate(self.random_effect_terms):
            slopes = slope_covariates[k]
            if slopes is None or len(np.atleast_1d(slopes)) == 0:
                z = np.array([1.0])
            else:
                slopes = np.atleast_1d(slopes)
                z = np.concatenate(([1.0], slopes))
                
            if len(z) != term.q:
                raise ValueError(f"Expected {term.q - 1} slope covariates for group {k+1}, got {len(z) - 1}.")
                
            cov += term.marginal_cov(z)
            
        return cov

    def summary(self) -> str:
        """
        Generate a comprehensive text summary of the fitted multivariate mixed effects model.
        
        Returns
        -------
        str
            A formatted string containing model metadata and variance estimates.
        """
        lines = []
        indent1 = "  "
        indent2 = "      "

        lines.append("\nMultivariate Mixed Effects Model Summary")
        lines.append("=" * 60)
        lines.append(indent1 + f"FE Model:             {type(self.fe_model).__name__}")
        lines.append(indent1 + f"Iterations:           {len(self.log_likelihood)}")
        status = "Yes (Early Stopped)" if self.is_early_stopped else ("Yes" if self.is_converged else "No")
        lines.append(indent1 + f"Converged:            {status}")
        lines.append(indent1 + f"Log-Likelihood:       {self.best_log_likelihood:.3f}")
        lines.append(indent1 + f"No. Outputs (m):      {self.m}")
        lines.append(indent1 + f"No. Grouping Factors: {self.k}")
        
        lines.append("-" * 60)
        lines.append(indent1 + "Unexplained Residual Variances")
        lines.append(indent2 + "{:<10} {:>12}".format("Response", "Variance"))
        for m in range(self.m):
            lines.append(indent2 + "{:<10} {:>12.4f}".format(m + 1, self.residual_term.cov[m, m]))
            
        lines.append("-" * 60)
        lines.append(indent1 + "Random Effects Variances (Diagonal)")
        lines.append(indent2 + "{:<8} {:<10} {:<15} {:>12}".format("Group", "Response", "Random Effect", "Variance"))
        
        for k, term in enumerate(self.random_effect_terms):
            q = term.q
            for i in range(self.m):
                for j in range(q):
                    idx = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    var = term.cov[idx, idx]
                    lines.append(indent2 + "{:<8} {:<10} {:<15} {:>12.4f}".format(k + 1, i + 1, effect_name, var))
                    
        lines.append("-" * 60)
        lines.append(indent1 + "* Note: To view full covariance/correlation matrices (including off-diagonals),")
        lines.append(indent1 + "  use `.residual_covariance`, `.random_effects_covariances`,")
        lines.append(indent1 + "  or their `_correlation` properties on this results object.")
        lines.append("=" * 60)
        
        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str


class EnsembleMixedEffectResults:
    """
    Deep Ensemble of MixedEffectResults models.
    
    Uses Welford's online algorithm and memory preallocation to compute epistemic 
    uncertainty (mean and standard deviation) with strictly O(1) memory scaling 
    with respect to the number of ensemble models.
    
    Attributes
    ----------
    results : list of MixedEffectResults
        The list of fitted single-seed model results.
    n_models : int
        The number of models in the ensemble.
    m : int
        Number of output responses.
    k : int
        Number of grouping factors.
    """
    def __init__(self, models: tuple[MixedEffectResults, ...]):
        if not models:
            raise ValueError("models cannot be empty.")
        self.models = models
        self.n_models = len(models)
        self.m = self.models[0].m
        self.k = self.models[0].k

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict using the ensemble with O(1) memory scaling.
        Uses Welford's algorithm to compute mean and standard deviation in-place.
        """
        n_samples = X.shape[0]
        
        # Initialize running aggregators
        mean_pred = np.zeros((n_samples, self.m), dtype=np.float64)
        M2_pred = np.zeros((n_samples, self.m), dtype=np.float64) # Sum of squares of differences
        
        for i, res in enumerate(self.models):
            pred = res.predict(X)
            
            # Welford's online variance calculation (In-place mutation)
            delta = pred - mean_pred
            mean_pred += delta / (i + 1)
            delta2 = pred - mean_pred
            M2_pred += delta * delta2
            
        # Population standard deviation (ddof=0 matches np.std)
        std_pred = np.sqrt(M2_pred / self.n_models)
        
        return mean_pred, std_pred

    def compute_random_effects(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple:
        """
        Estimate posterior random effects efficiently using Welford's algorithm.
        """
        # Run the first model to get exact array shapes and initialize accumulators
        # first sample and zero variance
        r0, tot0, mu0 = self.models[0].compute_random_effects(X, y, groups)
        
        mean_res, M2_res = r0.copy(), np.zeros_like(r0)
        mean_tot, M2_tot = tot0.copy(), np.zeros_like(tot0)
        
        mean_mu = [m.copy() for m in mu0]
        M2_mu = [np.zeros_like(m) for m in mu0]
        
        # Stream the remaining models
        for i in range(1, self.n_models):
            r, tot, mu = self.models[i].compute_random_effects(X, y, groups)
            
            # Accumulate Residuals
            delta_res = r - mean_res
            mean_res += delta_res / (i + 1)
            M2_res += delta_res * (r - mean_res)
            
            # Accumulate Total Effects
            delta_tot = tot - mean_tot
            mean_tot += delta_tot / (i + 1)
            M2_tot += delta_tot * (tot - mean_tot)
            
            # Accumulate Grouping Factors (mu)
            for k in range(self.k):
                delta_mu = mu[k] - mean_mu[k]
                mean_mu[k] += delta_mu / (i + 1)
                M2_mu[k] += delta_mu * (mu[k] - mean_mu[k])
                
        # Finalize standard deviations
        std_res = np.sqrt(M2_res / self.n_models)
        std_tot = np.sqrt(M2_tot / self.n_models)
        std_mu = tuple(np.sqrt(M2 / self.n_models) for M2 in M2_mu)
        
        return mean_res, std_res, mean_tot, std_tot, tuple(mean_mu), std_mu

    @property
    def residual_covariance(self) -> np.ndarray:
        """Expected residual covariance matrix (Memory Preallocated)."""
        covs = np.empty((self.n_models, self.m, self.m), dtype=np.float64)
        for i, res in enumerate(self.models):
            covs[i] = res.residual_covariance
        return np.mean(covs, axis=0)

    @property
    def residual_covariance_std(self) -> np.ndarray:
        """Epistemic uncertainty of residual covariance matrix."""
        covs = np.empty((self.n_models, self.m, self.m), dtype=np.float64)
        for i, res in enumerate(self.models):
            covs[i] = res.residual_covariance
        return np.std(covs, axis=0)

    @property
    def random_effects_covariances(self) -> tuple[np.ndarray]:
        """Expected covariance matrices for each grouping factor."""
        avg_covs = []
        for k in range(self.k):
            q = self.models[0].random_effect_terms[k].q
            covs = np.empty((self.n_models, q * self.m, q * self.m), dtype=np.float64)
            for i, res in enumerate(self.models):
                covs[i] = res.random_effects_covariances[k]
            avg_covs.append(np.mean(covs, axis=0))
        return avg_covs

    @property
    def random_effects_covariances_std(self) -> tuple[np.ndarray]:
        """Epistemic uncertainty of the covariance matrices."""
        std_covs = []
        for k in range(self.k):
            q = self.models[0].random_effect_terms[k].q
            covs = np.empty((self.n_models, q * self.m, q * self.m), dtype=np.float64)
            for i, res in enumerate(self.models):
                covs[i] = res.random_effects_covariances[k]
            std_covs.append(np.std(covs, axis=0))
        return std_covs

    def get_marginal_covariance(self, slope_covariates: tuple[np.ndarray] = None) -> np.ndarray:
        """Compute the expected total marginal covariance matrix."""
        covs = np.empty((self.n_models, self.m, self.m), dtype=np.float64)
        for i, res in enumerate(self.models):
            covs[i] = res.get_marginal_covariance(slope_covariates)
        return np.mean(covs, axis=0)

    def get_marginal_covariance_std(self, slope_covariates: tuple[np.ndarray] = None) -> np.ndarray:
        """Compute the epistemic uncertainty of the marginal covariance matrix."""
        covs = np.empty((self.n_models, self.m, self.m), dtype=np.float64)
        for i, res in enumerate(self.models):
            covs[i] = res.get_marginal_covariance(slope_covariates)
        return np.std(covs, axis=0)

    def summary(self) -> str:
        """Generate a comprehensive text summary of the ensemble model."""
        lines = []
        indent1 = "  "
        indent2 = "      "

        lines.append("\nDeep Ensemble Mixed Effects Model Summary")
        lines.append("=" * 60)
        lines.append(indent1 + f"Ensemble Size:          {self.n_models} models")
        lines.append(indent1 + f"No. Outputs (m):        {self.m}")
        lines.append(indent1 + f"No. Grouping Factors:   {self.k}")
        
        avg_ll = np.mean([res.best_log_likelihood for res in self.models])
        lines.append(indent1 + f"Expected Log-Likelihood:{avg_ll:.3f}")
        
        lines.append("-" * 60)
        lines.append(indent1 + "Expected Unexplained Residual Variances (Diagonal)")
        lines.append(indent2 + "{:<10} {:>12} {:>12}".format("Response", "Mean Var", "Epistemic SD"))
        
        mean_res_cov = self.residual_covariance
        std_res_cov = self.residual_covariance_std
        for m in range(self.m):
            lines.append(indent2 + "{:<10} {:>12.4f} {:>12.4f}".format(
                m + 1, mean_res_cov[m, m], std_res_cov[m, m]))
            
        lines.append("-" * 60)
        lines.append(indent1 + "Expected Random Effects Variances (Diagonal)")
        lines.append(indent2 + "{:<8} {:<10} {:<15} {:>12} {:>12}".format(
            "Group", "Response", "Random Effect", "Mean Var", "Epistemic SD"))
        
        mean_re_covs = self.random_effects_covariances
        std_re_covs = self.random_effects_covariances_std
        
        for k in range(self.k):
            q = self.models[0].random_effect_terms[k].q
            for i in range(self.m):
                for j in range(q):
                    idx = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    mean_var = mean_re_covs[k][idx, idx]
                    std_var = std_re_covs[k][idx, idx]
                    lines.append(indent2 + "{:<8} {:<10} {:<15} {:>12.4f} {:>12.4f}".format(
                        k + 1, i + 1, effect_name, mean_var, std_var))
                    
        lines.append("=" * 60)
        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str