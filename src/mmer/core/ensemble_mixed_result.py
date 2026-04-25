import numpy as np
from .mixed_result import MixedEffectResults

class EnsembleMixedEffectResults:
    """
    Deep Ensemble of MixedEffectResults models.

    Aggregates predictions and variance estimates across an ensemble of
    independently fitted ``MixedEffectResults``.

    It uses Welford's online algorithm to compute ensemble means and epistemic standard deviations
    for predictions, residuals, and covariance matrices in a single pass through the models.

    Parameters
    ----------
    models : tuple of MixedEffectResults
        Fitted single-seed model results. Must be non-empty and all models
        must share the same ``m`` and ``k``.

    Attributes
    ----------
    models : tuple of MixedEffectResults
        The constituent fitted model results.
    n_models : int
        Number of models in the ensemble.
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

    def __len__(self) -> int:
        return self.n_models

    def __iter__(self):
        return iter(self.models)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Expected value and epistemic std of the population prediction across the ensemble.
        Predicts the fixed effects component only, without any random effects.

        Parameters
        ----------
        X : np.ndarray
            Covariates, shape ``(n, p)``.

        Returns
        -------
        mean_pred : np.ndarray
            Ensemble mean prediction, shape ``(n, m)``.
        std_pred : np.ndarray
            Ensemble population standard deviation, shape ``(n, m)``.
        """
        n_samples = X.shape[0]
        mean_pred = np.zeros((n_samples, self.m), dtype=np.float64)
        M2_pred   = np.zeros((n_samples, self.m), dtype=np.float64)

        for i, res in enumerate(self.models):
            pred = res.predict(X)
            if pred.ndim == 1:           # guard for m == 1 models
                pred = pred[:, None]
            delta       = pred - mean_pred
            mean_pred  += delta / (i + 1)
            M2_pred    += delta * (pred - mean_pred)

        std_pred = np.sqrt(M2_pred / self.n_models)
        return mean_pred, std_pred

    def compute_random_effects(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple:
        """
        Estimate expected value and epistemic std of the residuals, total random effect,
        and posterior random effects across the ensemble.

        Parameters
        ----------
        X : np.ndarray
            Features, shape ``(n, p)``.
        y : np.ndarray
            Targets, shape ``(n, m)``.
        groups : np.ndarray
            Grouping factors, shape ``(n, k)``.

        Returns
        -------
        mean_res : np.ndarray
            Mean residuals after accounting for all effects, shape ``(n, m)``.
        std_res : np.ndarray
            Epistemic std of residuals, shape ``(n, m)``.
        mean_tot : np.ndarray
            Mean total random effect, shape ``(n, m)``.
        std_tot : np.ndarray
            Epistemic std of total random effect, shape ``(n, m)``.
        mean_mu : tuple of np.ndarray
            Mean posterior random effects per grouping factor,
            each of shape ``(groups, m, q)``.
        std_mu : tuple of np.ndarray
            Epistemic std of posterior random effects per grouping factor.
        """
        # Bootstrap from the first model to obtain shapes, then accumulate
        r0, tot0, mu0 = self.models[0].compute_random_effects(X, y, groups)

        mean_res, M2_res = r0.copy(),   np.zeros_like(r0)
        mean_tot, M2_tot = tot0.copy(), np.zeros_like(tot0)
        mean_mu = [m.copy() for m in mu0]
        M2_mu   = [np.zeros_like(m) for m in mu0]

        for i in range(1, self.n_models):
            r, tot, mu = self.models[i].compute_random_effects(X, y, groups)

            delta_res  = r - mean_res
            mean_res  += delta_res / (i + 1)
            M2_res    += delta_res * (r - mean_res)

            delta_tot  = tot - mean_tot
            mean_tot  += delta_tot / (i + 1)
            M2_tot    += delta_tot * (tot - mean_tot)

            for k in range(self.k):
                delta_mu      = mu[k] - mean_mu[k]
                mean_mu[k]   += delta_mu / (i + 1)
                M2_mu[k]     += delta_mu * (mu[k] - mean_mu[k])

        std_res = np.sqrt(M2_res / self.n_models)
        std_tot = np.sqrt(M2_tot / self.n_models)
        std_mu  = tuple(np.sqrt(M2 / self.n_models) for M2 in M2_mu)

        return mean_res, std_res, mean_tot, std_tot, tuple(mean_mu), std_mu

    # ------------------------------------------------------------------
    # Private Welford helpers — compute (mean, std) in a single pass
    # ------------------------------------------------------------------

    def _welford_residual_cov(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble mean and std of the residual covariance in one pass.

        Returns
        -------
        mean : np.ndarray, shape (m, m)
        std  : np.ndarray, shape (m, m)
        """
        mean = np.zeros((self.m, self.m), dtype=np.float64)
        M2   = np.zeros((self.m, self.m), dtype=np.float64)
        for i, res in enumerate(self.models):
            x      = res.residual_covariance
            delta  = x - mean
            mean  += delta / (i + 1)
            M2    += delta * (x - mean)
        return mean, np.sqrt(M2 / self.n_models)

    def _welford_re_covs(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute ensemble mean and std of every RE covariance matrix in one pass.

        Returns
        -------
        means : list of np.ndarray
            One ``(q*m, q*m)`` mean matrix per grouping factor.
        stds : list of np.ndarray
            One ``(q*m, q*m)`` std matrix per grouping factor.
        """
        qs    = [self.models[0].random_effect_terms[k].q for k in range(self.k)]
        dims  = [q * self.m for q in qs]
        means = [np.zeros((d, d), dtype=np.float64) for d in dims]
        M2s   = [np.zeros((d, d), dtype=np.float64) for d in dims]

        for i, res in enumerate(self.models):
            covs = res.random_effects_covariances
            for k in range(self.k):
                delta     = covs[k] - means[k]
                means[k] += delta / (i + 1)
                M2s[k]   += delta * (covs[k] - means[k])

        return means, [np.sqrt(M2 / self.n_models) for M2 in M2s]

    def _welford_marginal_cov(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble mean and std of the marginal covariance in one pass.

        Returns
        -------
        mean : np.ndarray, shape (m, m)
        std  : np.ndarray, shape (m, m)
        """
        mean = np.zeros((self.m, self.m), dtype=np.float64)
        M2   = np.zeros((self.m, self.m), dtype=np.float64)
        for i, res in enumerate(self.models):
            x      = res.get_marginal_covariance(slope_covariates)
            delta  = x - mean
            mean  += delta / (i + 1)
            M2    += delta * (x - mean)
        return mean, np.sqrt(M2 / self.n_models)

    # ------------------------------------------------------------------
    # Public covariance properties (delegate to single-pass helpers)
    # ------------------------------------------------------------------

    @property
    def residual_covariance(self) -> np.ndarray:
        """Ensemble mean of the residual covariance matrix, shape ``(m, m)``."""
        return self._welford_residual_cov()[0]

    @property
    def residual_covariance_std(self) -> np.ndarray:
        """Epistemic std of the residual covariance matrix, shape ``(m, m)``."""
        return self._welford_residual_cov()[1]

    @property
    def random_effects_covariances(self) -> list[np.ndarray]:
        """Ensemble mean covariance matrices for each grouping factor."""
        return self._welford_re_covs()[0]

    @property
    def random_effects_covariances_std(self) -> list[np.ndarray]:
        """Epistemic std of covariance matrices for each grouping factor."""
        return self._welford_re_covs()[1]

    def get_marginal_covariance(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Compute the ensemble mean of the total marginal covariance matrix.

        Parameters
        ----------
        slope_covariates : list of array-like or None, optional
            Random slope covariates per grouping factor (see
            ``MixedEffectResults.get_marginal_covariance``).

        Returns
        -------
        mean_cov : np.ndarray, shape ``(m, m)``
        """
        return self._welford_marginal_cov(slope_covariates)[0]

    def get_marginal_covariance_std(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Compute the epistemic std of the total marginal covariance matrix.

        Parameters
        ----------
        slope_covariates : list of array-like or None, optional
            Random slope covariates per grouping factor.

        Returns
        -------
        std_cov : np.ndarray, shape ``(m, m)``
        """
        return self._welford_marginal_cov(slope_covariates)[1]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Generate a text summary of the deep ensemble mixed effects model.

        Returns
        -------
        summary_str : str
            Formatted summary string (also printed to stdout).
        """
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

        # --- Residual covariance (single pass) ---
        mean_res_cov, std_res_cov = self._welford_residual_cov()

        lines.append("-" * 60)
        lines.append(indent1 + "Expected Unexplained Residual Variances (Diagonal)")
        lines.append(indent2 + "{:<10} {:>12} {:>12}".format("Response", "Mean Var", "Epistemic SD"))
        for m in range(self.m):
            lines.append(indent2 + "{:<10} {:>12.4f} {:>12.4f}".format(
                m + 1, mean_res_cov[m, m], std_res_cov[m, m]))

        # --- RE covariance (single pass) ---
        mean_re_covs, std_re_covs = self._welford_re_covs()

        lines.append("-" * 60)
        lines.append(indent1 + "Expected Random Effects Variances (Diagonal)")
        lines.append(indent2 + "{:<8} {:<10} {:<15} {:>12} {:>12}".format(
            "Group", "Response", "Random Effect", "Mean Var", "Epistemic SD"))

        for k in range(self.k):
            q = self.models[0].random_effect_terms[k].q
            for i in range(self.m):
                for j in range(q):
                    idx         = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    lines.append(indent2 + "{:<8} {:<10} {:<15} {:>12.4f} {:>12.4f}".format(
                        k + 1, i + 1, effect_name,
                        mean_re_covs[k][idx, idx], std_re_covs[k][idx, idx]))

        lines.append("=" * 60)
        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str