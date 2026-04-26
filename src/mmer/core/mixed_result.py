import numpy as np
from .mixed_effect import MixedEffectRegressor
from .inference import compute_random_effects_posterior
from .terms import RealizedRandomEffect, RealizedResidual


class MixedEffectResults:
    """
    MixedEffectRegressor fitted model results and inference interface.

    Provides convenient access to fitted model components, variance estimates,
    and methods for prediction and random effect estimation.

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
        realized_effects = tuple(
            RealizedRandomEffect(term, X, groups) for term in self.random_effect_terms
        )
        realized_residual = RealizedResidual(self.residual_term, n)

        # Predict Fixed Effects
        fx = self.fe_model.predict(X)
        fx = fx if self.m != 1 else fx[:, None]

        return compute_random_effects_posterior(
            realized_effects,
            realized_residual,
            y,
            fx,
            self.preconditioner,
            self.cg_maxiter,
        )

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

    def get_marginal_correlation(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
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

    def get_marginal_covariance(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
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
                raise ValueError(
                    f"Expected {term.q - 1} slope covariates for group {k+1}, got {len(z) - 1}."
                )

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
        status = (
            "Yes (Early Stopped)"
            if self.is_early_stopped
            else ("Yes" if self.is_converged else "No")
        )
        lines.append(indent1 + f"Converged:            {status}")
        lines.append(indent1 + f"Log-Likelihood:       {self.best_log_likelihood:.3f}")
        lines.append(indent1 + f"No. Outputs (m):      {self.m}")
        lines.append(indent1 + f"No. Grouping Factors: {self.k}")

        lines.append("-" * 60)
        lines.append(indent1 + "Unexplained Residual Variances")
        lines.append(indent2 + "{:<10} {:>12}".format("Response", "Variance"))
        for m in range(self.m):
            lines.append(
                indent2 + "{:<10} {:>12.4f}".format(m + 1, self.residual_term.cov[m, m])
            )

        lines.append("-" * 60)
        lines.append(indent1 + "Random Effects Variances (Diagonal)")
        lines.append(
            indent2
            + "{:<8} {:<10} {:<15} {:>12}".format(
                "Group", "Response", "Random Effect", "Variance"
            )
        )

        for k, term in enumerate(self.random_effect_terms):
            q = term.q
            for i in range(self.m):
                for j in range(q):
                    idx = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    var = term.cov[idx, idx]
                    lines.append(
                        indent2
                        + "{:<8} {:<10} {:<15} {:>12.4f}".format(
                            k + 1, i + 1, effect_name, var
                        )
                    )

        lines.append("-" * 60)
        lines.append(
            indent1
            + "* Note: To view full covariance/correlation matrices (including off-diagonals),"
        )
        lines.append(
            indent1 + "  use `.residual_covariance`, `.random_effects_covariances`,"
        )
        lines.append(
            indent1 + "  or their `_correlation` properties on this results object."
        )
        lines.append("=" * 60)

        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str
