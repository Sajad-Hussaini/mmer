import numpy as np
from .mixed_result import MixedEffectResults


class EnsembleMixedEffectResults:
    """
    Ensemble of independently fitted ``MixedEffectResults`` models.

    Aggregates predictions and uncertainty estimates across an ensemble of
    models trained with different random initialisations (seeds), providing
    both an ensemble mean and an element-wise epistemic standard deviation
    for each model output.

    **Epistemic uncertainty for matrix outputs**

    For covariance matrices, the mean and std are computed element-wise over
    the ensemble.  For correlation matrices, the mean is derived from the
    ensemble-mean covariance — ``cov_to_corr(mean_Σ)`` — because averaging
    correlations directly is incorrect: ``E[corr(Σᵢ)] ≠ corr(E[Σᵢ])``.
    The correlation std is the element-wise std of the per-model correlation
    matrices, which is a separate and independent quantity from the covariance std.

    Parameters
    ----------
    models : tuple of MixedEffectResults
        Fitted model results. Must be non-empty; all models must share the
        same ``m`` and ``k``.

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
        self.m = models[0].m
        self.k = models[0].k

    def __len__(self) -> int:
        return self.n_models

    def __iter__(self):
        return iter(self.models)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensemble mean and epistemic std of the fixed-effects prediction.

        The epistemic std reflects model uncertainty due to different random
        initialisations and does not include aleatoric (residual) uncertainty.

        Parameters
        ----------
        X : np.ndarray
            Covariates, shape ``(n, p)``.

        Returns
        -------
        mean_pred : np.ndarray
            Ensemble mean prediction, shape ``(n, m)``.
        std_pred : np.ndarray
            Epistemic std of the prediction, shape ``(n, m)``.
        """
        n = X.shape[0]
        mean = np.zeros((n, self.m), dtype=np.float64)
        M2 = np.zeros((n, self.m), dtype=np.float64)

        for i, res in enumerate(self.models):
            pred = res.predict(X)
            if pred.ndim == 1:
                pred = pred[:, None]
            delta = pred - mean
            mean += delta / (i + 1)
            M2 += delta * (pred - mean)

        return mean, np.sqrt(M2 / self.n_models)

    def compute_random_effects(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> tuple:
        """
        Ensemble mean and epistemic std of posterior random effects.

        Parameters
        ----------
        X : np.ndarray
            Covariates, shape ``(n, p)``.
        y : np.ndarray
            Targets, shape ``(n, m)``.
        groups : np.ndarray
            Grouping factor indicators, shape ``(n, k)``.

        Returns
        -------
        mean_res : np.ndarray
            Ensemble mean residuals, shape ``(n, m)``.
        std_res : np.ndarray
            Epistemic std of residuals, shape ``(n, m)``.
        mean_tot : np.ndarray
            Ensemble mean total random effect, shape ``(n, m)``.
        std_tot : np.ndarray
            Epistemic std of total random effect, shape ``(n, m)``.
        mean_mu : tuple of np.ndarray
            Ensemble mean posterior random effects, one array per grouping factor.
        std_mu : tuple of np.ndarray
            Epistemic std of posterior random effects, one array per grouping factor.
        """
        r0, tot0, mu0 = self.models[0].compute_random_effects(X, y, groups)

        mean_res = r0.astype(np.float64)
        M2_res = np.zeros_like(mean_res)
        mean_tot = tot0.astype(np.float64)
        M2_tot = np.zeros_like(mean_tot)
        mean_mu = [m.astype(np.float64) for m in mu0]
        M2_mu = [np.zeros_like(m, dtype=np.float64) for m in mu0]

        for i in range(1, self.n_models):
            r, tot, mu = self.models[i].compute_random_effects(X, y, groups)

            delta = r - mean_res
            mean_res += delta / (i + 1)
            M2_res += delta * (r - mean_res)

            delta = tot - mean_tot
            mean_tot += delta / (i + 1)
            M2_tot += delta * (tot - mean_tot)

            for k in range(self.k):
                delta = mu[k] - mean_mu[k]
                mean_mu[k] += delta / (i + 1)
                M2_mu[k] += delta * (mu[k] - mean_mu[k])

        std_res = np.sqrt(M2_res / self.n_models)
        std_tot = np.sqrt(M2_tot / self.n_models)
        std_mu = tuple(np.sqrt(M2 / self.n_models) for M2 in M2_mu)

        return mean_res, std_res, mean_tot, std_tot, tuple(mean_mu), std_mu

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _welford_matrix(self, values) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean and population std over an iterable of equal-shape arrays."""
        mean = None
        M2 = None
        n = 0
        for x in values:
            x = np.asarray(x, dtype=np.float64)
            if mean is None:
                mean = np.zeros_like(x)
                M2 = np.zeros_like(x)
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
        return mean, np.sqrt(M2 / n)

    def _welford_residual_cov(self) -> tuple[np.ndarray, np.ndarray]:
        """Welford mean and std of residual covariance across ensemble members."""
        return self._welford_matrix(res.residual_covariance for res in self.models)

    def _welford_residual_corr(self) -> tuple[np.ndarray, np.ndarray]:
        mean_cov, _ = self._welford_residual_cov()
        mean_corr = self.models[0].cov_to_corr(mean_cov)
        _, std_corr = self._welford_matrix(
            res.residual_correlation for res in self.models
        )
        return mean_corr, std_corr

    def _welford_re_covs(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Welford mean and std of RE covariance matrices across ensemble members."""
        shapes = [
            self.models[0].random_effects_covariances[k].shape for k in range(self.k)
        ]
        means = [np.zeros(s, dtype=np.float64) for s in shapes]
        M2s = [np.zeros(s, dtype=np.float64) for s in shapes]

        for i, res in enumerate(self.models):
            covs = res.random_effects_covariances
            for k in range(self.k):
                delta = covs[k] - means[k]
                means[k] += delta / (i + 1)
                M2s[k] += delta * (covs[k] - means[k])

        return means, [np.sqrt(M2 / self.n_models) for M2 in M2s]

    def _welford_re_corrs(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        mean_covs, _ = self._welford_re_covs()
        mean_corrs = [self.models[0].cov_to_corr(cov) for cov in mean_covs]

        shapes = [mean_covs[k].shape for k in range(self.k)]
        wf_means = [np.zeros(s, dtype=np.float64) for s in shapes]
        M2s = [np.zeros(s, dtype=np.float64) for s in shapes]

        for i, res in enumerate(self.models):
            corrs = res.random_effects_correlations
            for k in range(self.k):
                delta = corrs[k] - wf_means[k]
                wf_means[k] += delta / (i + 1)
                M2s[k] += delta * (corrs[k] - wf_means[k])

        return mean_corrs, [np.sqrt(M2 / self.n_models) for M2 in M2s]

    def _welford_marginal_cov(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._welford_matrix(
            res.get_marginal_covariance(slope_covariates) for res in self.models
        )

    def _welford_marginal_corr(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_cov, _ = self._welford_marginal_cov(slope_covariates)
        mean_corr = self.models[0].cov_to_corr(mean_cov)
        _, std_corr = self._welford_matrix(
            res.get_marginal_correlation(slope_covariates) for res in self.models
        )
        return mean_corr, std_corr

    # ------------------------------------------------------------------
    # Public properties — covariances
    # ------------------------------------------------------------------

    @property
    def residual_covariance(self) -> np.ndarray:
        """Ensemble mean residual covariance matrix, shape ``(m, m)``."""
        return self._welford_residual_cov()[0]

    @property
    def residual_covariance_std(self) -> np.ndarray:
        """
        Epistemic std of the residual covariance matrix, shape ``(m, m)``.

        Each entry is the std of that covariance element across ensemble members.
        """
        return self._welford_residual_cov()[1]

    @property
    def random_effects_covariances(self) -> list[np.ndarray]:
        """
        Ensemble mean RE covariance matrix for each grouping factor.

        Returns
        -------
        list of np.ndarray
            One matrix of shape ``(q*m, q*m)`` per grouping factor.
        """
        return self._welford_re_covs()[0]

    @property
    def random_effects_covariances_std(self) -> list[np.ndarray]:
        """
        Epistemic std of the RE covariance matrix for each grouping factor.

        Each entry is the std of that covariance element across ensemble members.

        Returns
        -------
        list of np.ndarray
            One matrix of shape ``(q*m, q*m)`` per grouping factor.
        """
        return self._welford_re_covs()[1]

    # ------------------------------------------------------------------
    # Public properties — correlations
    # ------------------------------------------------------------------

    @property
    def residual_correlation(self) -> np.ndarray:
        """
        Ensemble mean residual correlation matrix, shape ``(m, m)``.

        Derived from the ensemble-mean covariance as ``cov_to_corr(mean_Σ)``,
        which is the correct estimator.  Averaging per-model correlations
        directly is incorrect because the covariance-to-correlation transform
        is nonlinear.
        """
        return self._welford_residual_corr()[0]

    @property
    def residual_correlation_std(self) -> np.ndarray:
        """
        Epistemic std of the residual correlation matrix, shape ``(m, m)``.

        Each entry is the std of that correlation element across ensemble members.
        This is an independent quantity from the covariance std and can be used
        to place uncertainty bands on individual correlation values (e.g., when
        plotting inter-period SA correlations).
        """
        return self._welford_residual_corr()[1]

    @property
    def random_effects_correlations(self) -> list[np.ndarray]:
        """
        Ensemble mean RE correlation matrix for each grouping factor.

        Derived from the ensemble-mean covariance (see ``residual_correlation``).

        Returns
        -------
        list of np.ndarray
            One matrix of shape ``(q*m, q*m)`` per grouping factor.
        """
        return self._welford_re_corrs()[0]

    @property
    def random_effects_correlations_std(self) -> list[np.ndarray]:
        """
        Epistemic std of the RE correlation matrix for each grouping factor.

        Each entry is the std of that correlation element across ensemble members.

        Returns
        -------
        list of np.ndarray
            One matrix of shape ``(q*m, q*m)`` per grouping factor.
        """
        return self._welford_re_corrs()[1]

    # ------------------------------------------------------------------
    # Public methods — marginal covariance / correlation
    # ------------------------------------------------------------------

    def get_marginal_covariance(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Ensemble mean of the total marginal covariance matrix, shape ``(m, m)``.

        The marginal covariance combines residual and random-effect contributions
        for a given covariate profile.

        Parameters
        ----------
        slope_covariates : tuple of np.ndarray or None, optional
            Random slope covariates for each grouping factor (exclude the intercept).
            Pass ``None`` to assume random-intercept-only models.
        """
        return self._welford_marginal_cov(slope_covariates)[0]

    def get_marginal_covariance_std(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Epistemic std of the total marginal covariance matrix, shape ``(m, m)``.

        Each entry is the std of that marginal covariance element across ensemble
        members.

        Parameters
        ----------
        slope_covariates : tuple of np.ndarray or None, optional
            Random slope covariates for each grouping factor (exclude the intercept).
            Pass ``None`` to assume random-intercept-only models.
        """
        return self._welford_marginal_cov(slope_covariates)[1]

    def get_marginal_correlation(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Ensemble mean of the total marginal correlation matrix, shape ``(m, m)``.

        Derived from the ensemble-mean marginal covariance (see
        ``residual_correlation`` for the mathematical rationale).

        Parameters
        ----------
        slope_covariates : tuple of np.ndarray or None, optional
            Random slope covariates for each grouping factor (exclude the intercept).
            Pass ``None`` to assume random-intercept-only models.
        """
        return self._welford_marginal_corr(slope_covariates)[0]

    def get_marginal_correlation_std(
        self, slope_covariates: tuple[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Epistemic std of the total marginal correlation matrix, shape ``(m, m)``.

        Each entry is the std of that marginal correlation element across ensemble
        members.

        Parameters
        ----------
        slope_covariates : tuple of np.ndarray or None, optional
            Random slope covariates for each grouping factor (exclude the intercept).
            Pass ``None`` to assume random-intercept-only models.
        """
        return self._welford_marginal_corr(slope_covariates)[1]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a text summary of the ensemble mixed effects model."""
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

        mean_res_cov, std_res_cov = self._welford_residual_cov()
        lines.append("-" * 60)
        lines.append(indent1 + "Expected Unexplained Residual Variances (Diagonal)")
        lines.append(
            indent2
            + "{:<10} {:>12} {:>12}".format("Response", "Mean Var", "Epistemic SD")
        )
        for m in range(self.m):
            lines.append(
                indent2
                + "{:<10} {:>12.4f} {:>12.4f}".format(
                    m + 1, mean_res_cov[m, m], std_res_cov[m, m]
                )
            )

        mean_re_covs, std_re_covs = self._welford_re_covs()
        lines.append("-" * 60)
        lines.append(indent1 + "Expected Random Effects Variances (Diagonal)")
        lines.append(
            indent2
            + "{:<8} {:<10} {:<15} {:>12} {:>12}".format(
                "Group", "Response", "Random Effect", "Mean Var", "Epistemic SD"
            )
        )

        for k in range(self.k):
            q = self.models[0].random_effect_terms[k].q
            for i in range(self.m):
                for j in range(q):
                    idx = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    lines.append(
                        indent2
                        + "{:<8} {:<10} {:<15} {:>12.4f} {:>12.4f}".format(
                            k + 1,
                            i + 1,
                            effect_name,
                            mean_re_covs[k][idx, idx],
                            std_re_covs[k][idx, idx],
                        )
                    )

        lines.append("=" * 60)
        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str
