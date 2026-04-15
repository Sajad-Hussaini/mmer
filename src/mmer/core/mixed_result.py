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
    def __init__(self, mixed_model: MixedEffectRegressor):
        # Expose convenient attributes directly, decoupling from regressor model state
        self.fe_model = mixed_model.fe_model
        self.m = mixed_model.m
        self.k = mixed_model.k
        self.random_effect_terms = mixed_model.random_effect_terms
        self.residual_term = mixed_model.residual_term
        self.log_likelihood = mixed_model.log_likelihood
        self.is_converged = mixed_model._is_converged
        self.best_log_likelihood = mixed_model._best_log_likelihood
        self.preconditioner = mixed_model.preconditioner

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
        >>> y_pred = model.predict(X_new)
        
        >>> # Predict with both fixed and random effects
        >>> y_fixed = model.predict(X_new)
        >>> _, total_re, _ = model.compute_random_effects(X_new, y_new, groups_new)
        >>> y_pred_full = y_fixed + total_re.reshape((-1, model.m))
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
        
        return compute_random_effects_posterior(
            realized_effects, realized_residual, y, fx,
            self.random_effect_terms, self.preconditioner
        )

    @property
    def residual_covariance(self) -> np.ndarray:
        """Get the estimated residual covariance matrix."""
        return self.residual_term.cov
        
    @property
    def random_effects_covariances(self) -> list[np.ndarray]:
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
    def random_effects_correlations(self) -> list[np.ndarray]:
        """Get the estimated correlation matrices for each random effect grouping factor."""
        return [self.cov_to_corr(cov) for cov in self.random_effects_covariances]

    def get_marginal_correlation(self, slope_covariates: list[np.ndarray | None] | None = None) -> np.ndarray:
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

    def get_marginal_covariance(self, slope_covariates: list[np.ndarray | None] | None = None) -> np.ndarray:
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

    def summary(self):
        """
        Display a summary of the fitted multivariate mixed effects model.
        """
        indent0 = ""
        indent1 = "   "
        indent2 = "       "

        print("\n" + indent0 + "Multivariate Mixed Effects Model Summary")
        print("=" * 50)
        print(indent1 + f"FE Model: {type(self.fe_model).__name__}")
        print(indent1 + f"Iterations: {len(self.log_likelihood)}")
        print(indent1 + f"Converged: {self.is_converged}")
        print(indent1 + f"Log-Likelihood: {self.best_log_likelihood:.3f}")
        print(indent1 + f"No. Outputs: {self.m}")
        print(indent1 + f"No. Grouping Factors: {self.k}")
        print("-" * 50)
        print(indent1 + f"Unexplained Residual Variances")
        print(indent2 + "{:<10} {:>10}".format("Response", "Variance"))
        for m in range(self.m):
            print(indent2 + "{:<10} {:>10.4f}".format(m + 1, self.residual_term.cov[m, m]))
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        print(indent2 + "{:<8} {:<10} {:<15} {:>10}".format("Group", "Response", "Random Effect", "Variance"))
        
        for k, term in enumerate(self.random_effect_terms):
            # q is term.q (1 + number of slopes)
            # term.cov is (m*q, m*q)
            q = term.q
            for i in range(self.m):
                for j in range(q):
                    idx = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    # Access diagonal element
                    var = term.cov[idx, idx]
                    print(indent2 + "{:<8} {:<10} {:<15} {:>10.4f}".format(k + 1, i + 1, effect_name, var))
        print("\n")