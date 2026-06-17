import numpy as np
from .result import MixedModel
from .posteriors import Estimate, ObservationPosterior, GroupPosterior, InferenceResult


class EnsembleMixedModel:
    """
    Ensemble Wrapper for Independently Fitted Mixed-Effects Models.

    Aggregates multiple independently fitted `MixedModel` instances to quantify
    epistemic uncertainty (model uncertainty) in predictions, group effects, and
    variance components.

    Parameters
    ----------
    models : list of MixedModel
        The collection of previously fitted `MixedModel` instances.

    Attributes
    ----------
    models : list of MixedModel
        The collection of previously fitted `MixedModel` instances.
    n_models : int
        The number of sub-models in the ensemble.
    n_responses : int
        The number of response variables.
    n_groups : int
        The number of grouping factors.
    """

    def __init__(self, models: list[MixedModel]):
        if not models:
            raise ValueError("models list cannot be empty.")
        self.models = models
        self.n_models = len(models)
        self.n_responses = models[0].n_responses
        self.n_groups = models[0].n_groups

    def predict(self, X: np.ndarray, groups: np.ndarray = None) -> Estimate:
        """
        Aggregate predictions across all ensemble models, incorporating personalized random effects if known.

        This method works identically to `MixedModel.predict`, but it evaluates the test data against every
        individual model in the ensemble. It then aggregates the point predictions to form the expected value,
        and computes the standard deviation across models to quantify epistemic uncertainty.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The fixed-effects design matrix.
        groups : ndarray of shape (n_samples, n_groups), optional
            The categorical grouping factors. If provided, models will attempt a dictionary lookup
            to add their individually learned Best Linear Unbiased Predictors (BLUPs) to the prediction.

        Returns
        -------
        Estimate
            An estimate object containing:
            - `value`: The expected (mean) prediction across the ensemble.
            - `std`: The epistemic standard deviation representing model disagreement.
        """
        preds = np.stack([m.predict(X, groups=groups) for m in self.models], axis=0)
        return Estimate(
            value=np.mean(preds, axis=0),
            std=np.std(preds, axis=0, ddof=1 if self.n_models > 1 else 0),
        )

    @property
    def R(self) -> Estimate:
        """The expected residual covariance matrix and its epistemic standard deviation."""
        covs = np.stack([m.R for m in self.models], axis=0)
        return Estimate(
            value=np.mean(covs, axis=0),
            std=np.std(covs, axis=0, ddof=1 if self.n_models > 1 else 0),
        )

    @property
    def G(self) -> list[Estimate]:
        """The expected covariance matrix and epistemic standard deviation for each group."""
        results = []
        for k in range(self.n_groups):
            covs = np.stack([m.G[k] for m in self.models], axis=0)
            results.append(
                Estimate(
                    value=np.mean(covs, axis=0),
                    std=np.std(covs, axis=0, ddof=1 if self.n_models > 1 else 0),
                )
            )
        return results

    def infer(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> InferenceResult:
        """
        Compute the posterior estimates of residuals and random effects across the entire ensemble.

        Unlike `predict`, which uses historically learned BLUPs, this method assumes the true outcomes
        ``y`` are known. It passes the dataset through each base model's `infer` engine to calculate
        the exact structural residuals and optimal random effects. It then aggregates these properties
        across the ensemble to quantify both the expected analytical posteriors and their epistemic uncertainties.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The fixed-effects design matrix.
        y : ndarray of shape (n_samples, n_responses)
            The true continuous response matrix.
        groups : ndarray of shape (n_samples, n_groups)
            The categorical grouping factors.

        Returns
        -------
        InferenceResult
            A unified container holding the aggregated residuals, total random effects,
            and specific group BLUPs. Every numeric property inside this result is an `Estimate`
            object containing both the expected `value` and the epistemic `std` deviation.
        """
        resid_list = []
        tot_re_list = []
        mu_lists = [[] for _ in range(self.n_groups)]

        group_levels = {}
        group_counts = {}

        for i, m in enumerate(self.models):
            posterior = m.infer(X, y, groups)
            resid_list.append(posterior.observations.residuals.value)
            tot_re_list.append(posterior.observations.total_random_effects.value)
            for k in range(self.n_groups):
                mu_lists[k].append(posterior.groups[k].effects.value)
                if i == 0:
                    group_levels[k] = posterior.groups[k].levels
                    group_counts[k] = posterior.groups[k].counts

        resid_stack = np.stack(resid_list, axis=0)
        tot_re_stack = np.stack(tot_re_list, axis=0)

        resid_mean = np.mean(resid_stack, axis=0)
        resid_std = np.std(resid_stack, axis=0, ddof=1 if self.n_models > 1 else 0)

        tot_re_mean = np.mean(tot_re_stack, axis=0)
        tot_re_std = np.std(tot_re_stack, axis=0, ddof=1 if self.n_models > 1 else 0)

        group_posteriors = []
        for k in range(self.n_groups):
            mu_stack = np.stack(mu_lists[k], axis=0)
            group_posteriors.append(
                GroupPosterior(
                    levels=group_levels[k],
                    counts=group_counts[k],
                    effects=Estimate(
                        value=np.mean(mu_stack, axis=0),
                        std=np.std(
                            mu_stack, axis=0, ddof=1 if self.n_models > 1 else 0
                        ),
                    ),
                )
            )

        return InferenceResult(
            observations=ObservationPosterior(
                residuals=Estimate(value=resid_mean, std=resid_std),
                total_random_effects=Estimate(value=tot_re_mean, std=tot_re_std),
            ),
            groups=group_posteriors,
        )

    def marginal_cov(
        self, slope_covariates: list[np.ndarray | None] | None = None
    ) -> Estimate:
        """
        Compute the expected unconditional marginal covariance matrix.

        Parameters
        ----------
        slope_covariates : list of np.ndarray or None, optional
            A list specifying the covariates (excluding the intercept) for each grouping
            factor.

        Returns
        -------
        Estimate
            An estimate object containing the expected covariance matrix and its
            epistemic standard deviation.
        """
        covs = np.stack([m.marginal_cov(slope_covariates) for m in self.models], axis=0)
        return Estimate(
            value=np.mean(covs, axis=0),
            std=np.std(covs, axis=0, ddof=1 if self.n_models > 1 else 0),
        )

    def marginal_corr(
        self, slope_covariates: list[np.ndarray | None] | None = None
    ) -> Estimate:
        """
        Compute the expected unconditional marginal correlation matrix.

        Parameters
        ----------
        slope_covariates : list of np.ndarray or None, optional
            A list specifying the covariates (excluding the intercept) for each grouping
            factor.

        Returns
        -------
        Estimate
            An estimate object containing the expected correlation matrix and its
            epistemic standard deviation.
        """
        corrs = np.stack(
            [m.marginal_corr(slope_covariates) for m in self.models], axis=0
        )
        return Estimate(
            value=np.mean(corrs, axis=0),
            std=np.std(corrs, axis=0, ddof=1 if self.n_models > 1 else 0),
        )

    def summary(self) -> str:
        """
        Generate a formatted summary string of the ensemble model.

        Returns
        -------
        str
            A human-readable text table summarizing the expected variance components
            and their epistemic standard deviations.
        """
        lines = []
        indent1 = "  "
        indent2 = "      "

        lines.append("\nEnsemble Mixed Effects Model Summary")
        lines.append("=" * 70)
        lines.append(indent1 + f"Ensemble Size:          {self.n_models} models")
        lines.append(indent1 + f"Outputs (n_responses):  {self.n_responses}")
        lines.append(indent1 + f"Groups (n_groups):      {self.n_groups}")
        avg_ll = np.mean([res.best_log_likelihood for res in self.models])
        lines.append(indent1 + f"Expected Log-Likelihood:{avg_ll:.3f}")

        lines.append("-" * 70)
        lines.append(indent1 + "Expected Residual Variances (R)")
        lines.append(
            indent2
            + "{:<10} {:>12} {:>12}".format("Response", "Mean Var", "Epistemic Std")
        )
        for i in range(self.n_responses):
            lines.append(
                indent2
                + "{:<10} {:>12.4f} {:>12.4f}".format(
                    i + 1, self.R.value[i, i], self.R.std[i, i]
                )
            )

        lines.append("-" * 60)
        lines.append(indent1 + "Random Effects Variances (G)")
        lines.append(
            indent2
            + "{:<8} {:<10} {:<15} {:>12} {:>12}".format(
                "Group", "Response", "Effect", "E[Var]", "Std[Var]"
            )
        )

        for k, cov in enumerate(self.G):
            # To get q, we need to inspect the base model's G
            base_cov = self.models[0]._G[k]
            q = base_cov.n_effects
            for i in range(self.n_responses):
                for j in range(q):
                    idx = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    lines.append(
                        indent2
                        + "{:<8} {:<10} {:<15} {:>12.4f} {:>12.4f}".format(
                            k + 1,
                            i + 1,
                            effect_name,
                            cov.value[idx, idx],
                            cov.std[idx, idx],
                        )
                    )

        lines.append("=" * 70)
        summary_str = "\n".join(lines)
        return summary_str

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"<EnsembleMixedModel: {self.n_models} models, {self.n_responses} responses, {self.n_groups} groups>"
