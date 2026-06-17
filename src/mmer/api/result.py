import numpy as np
import pandas as pd
from ..structures.inference import compute_random_effects_posterior
from ..structures.covariance import RandomCovariance, ResidualCovariance
from ..structures.design import RandomDesign
from .posteriors import Estimate, ObservationPosterior, GroupPosterior, InferenceResult


class MixedModel:
    """
    Fitted Mixed-Effects Model Result.

    Encapsulates the fitted fixed-effects model alongside the learned variance
    components (covariance matrices). Provides access to model selection metrics
    (AIC/BIC), prediction, random effects inference (BLUPs), and diagnostics.

    Attributes
    ----------
    fixed_effects_model : object
        The fitted base predictive model containing the fixed effects.
    n_samples : int
        The number of samples used during fitting.
    n_responses : int
        The number of response variables (outputs).
    n_groups : int
        The number of grouping factors.
    G : list of np.ndarray
        The learned random effects covariance matrices, one for each grouping factor.
    R : np.ndarray
        The learned residual covariance matrix.
    log_likelihood : list of float
        The history of the log-likelihood over the EM iterations.
    is_converged : bool
        Whether the EM algorithm successfully reached the tolerance limit.
    best_log_likelihood : float
        The highest log-likelihood achieved during fitting.
    """

    def __init__(
        self,
        fixed_effects_model,
        n_samples: int,
        n_responses: int,
        n_groups: int,
        G: tuple[RandomCovariance, ...],
        R: ResidualCovariance,
        log_likelihood: list[float],
        is_converged: bool,
        is_early_stopped: bool,
        best_log_likelihood: float,
        preconditioner: bool,
        cg_maxiter: int,
        force_iterative: bool,
        random_slopes: tuple,
        training_group_posteriors: list[GroupPosterior] = None,
    ):
        self.fixed_effects_model = fixed_effects_model
        self.n_samples = n_samples
        self.n_responses = n_responses
        self.n_groups = n_groups
        self._G = G
        self._R = R
        self.log_likelihood = log_likelihood
        self.is_converged = is_converged
        self.is_early_stopped = is_early_stopped
        self.best_log_likelihood = best_log_likelihood
        self.preconditioner = preconditioner
        self.cg_maxiter = cg_maxiter
        self.force_iterative = force_iterative
        self.random_slopes = random_slopes
        self.training_group_posteriors = training_group_posteriors

    def predict(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        """
        Predict responses using the fixed-effects model, and optionally include personalized random effects.

        By default, if `groups` is not provided, predictions strictly reflect population-level trends
        learned by the base estimator. If `groups` are provided, the model performs a fast dictionary-style
        lookup to match the input groups to the properties (BLUPs) it learned during training.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The fixed-effects design matrix (covariates).
        groups : ndarray of shape (n_samples, n_groups), optional
            The categorical grouping factors for each observation. If provided, the predictions
            will be personalized by adding the learned random effect offsets.

        Returns
        -------
        ndarray of shape (n_samples, n_responses)
            The predicted response values.

        Notes
        -----
            **Known vs. Unknown Groups:**
            When `groups` are provided, the prediction logic matches each group identifier to the
            effects learned during training.
            - If a group was **known** (seen during training), its learned Best Linear Unbiased Predictor
            (BLUP) offset is retrieved, scaled by any random slope covariates, and added to the prediction.
            - If a group is **unknown** (not present in the training data), the lookup safely defaults
            the random effect to 0. This mathematically grounds the prediction back to the population average.

        Importantly, this method does *not* recalculate new random effects or residuals based on the
        input `X`. It purely applies historically learned offsets. For evaluating exact posterior
        properties on a validation dataset where true outcomes are known, use :meth:`infer` instead.
        """
        pred = self.fixed_effects_model.predict(X)
        if self.n_responses == 1 and pred.ndim == 1:
            pred = pred[:, None]

        if groups is not None:
            if groups.ndim == 1:
                groups = groups[:, None]
            if groups.shape[1] != self.n_groups:
                raise ValueError(
                    f"Expected {self.n_groups} columns in groups, but got {groups.shape[1]}"
                )
            if getattr(self, "training_group_posteriors", None) is None:
                raise ValueError(
                    "Model was not fitted with training groups stored. Cannot predict with personalized random effects."
                )

            re_pred = np.zeros_like(pred)
            for k in range(self.n_groups):
                test_levels = groups[:, k]
                train_levels = self.training_group_posteriors[k].levels
                train_effects = self.training_group_posteriors[k].effects.value

                slope_cols = self.random_slopes[k]
                if slope_cols is None:
                    z_i = np.ones((X.shape[0], 1), dtype=X.dtype)
                else:
                    z_i = np.concatenate(
                        [np.ones((X.shape[0], 1), dtype=X.dtype), X[:, slope_cols]],
                        axis=1,
                    )

                idx = np.searchsorted(train_levels, test_levels)
                # Handle elements not found or out of bounds
                idx[idx == len(train_levels)] = 0
                found = train_levels[idx] == test_levels

                sample_effects = np.zeros((
                    X.shape[0],
                    self.n_responses,
                    train_effects.shape[2],
                ))
                sample_effects[found] = train_effects[idx[found]]

                # Add to total prediction using Einstein summation
                re_pred += np.einsum("nq,nmq->nm", z_i, sample_effects)

            pred += re_pred

        return pred

    def infer(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Perform exact posterior inference on a specific dataset where true outcomes are known.

        Unlike :meth:`predict`, which applies historically learned BLUPs to new data, `infer`
        takes the true known responses (``y``) and mathematically deconstructs them. It solves
        the generalized least squares equations strictly for the provided dataset using the population-level
        predictions and the globally learned variance matrices (``G`` and ``R``).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The fixed-effects design matrix.
        y : ndarray of shape (n_samples, n_responses)
            The true, known continuous response matrix.
        groups : ndarray of shape (n_samples, n_groups)
            The grouping factors linking observations to group structures.

        Returns
        -------
        InferenceResult
            A unified structural container holding the exactly reconstructed observation-level
            properties (conditional residuals and total random effects) and group-level
            properties (the mathematically optimal BLUP offsets for these specific groups).

        Notes
        -----
            **When to use `infer` vs `predict`:**
            - Use **`predict(X, groups)`** in production when ``y`` is entirely unknown and you simply want to
            guess the outcome using previously learned group profiles.
            - Use **`infer(X, y, groups)`** during validation, testing, or model diagnostics. Because ``y``
            is known, `infer` ignores the old training BLUPs and precisely calculates exactly
            how much of the error was driven by group properties versus pure noise in this specific dataset.
        """
        if y.ndim == 1:
            y = y[:, None]
        if groups.ndim == 1:
            groups = groups[:, None]
        if groups.shape[1] != self.n_groups:
            raise ValueError(
                f"Expected {self.n_groups} columns in groups, but got {groups.shape[1]}"
            )

        random_designs = []
        for i, slope_cols in enumerate(self.random_slopes):
            covariates = X[:, slope_cols] if slope_cols is not None else None
            random_designs.append(RandomDesign(groups[:, i], covariates))

        random_designs = tuple(random_designs)

        fixed_effects_predictions = self.predict(X)

        resid, total_re, mu_list = compute_random_effects_posterior(
            self._G,
            random_designs,
            self._R,
            y,
            fixed_effects_predictions,
            self.preconditioner,
            self.cg_maxiter,
            self.force_iterative,
        )

        group_posteriors = []
        for k in range(self.n_groups):
            levels, counts = np.unique(groups[:, k], return_counts=True)
            group_posteriors.append(
                GroupPosterior(
                    levels=levels, counts=counts, effects=Estimate(value=mu_list[k])
                )
            )

        return InferenceResult(
            observations=ObservationPosterior(
                residuals=Estimate(value=resid),
                total_random_effects=Estimate(value=total_re),
            ),
            groups=group_posteriors,
        )

    def residuals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        type: str = "conditional",
    ) -> np.ndarray:
        """
        Calculate residuals of the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The fixed-effects design matrix.
        y : np.ndarray
            The true response matrix.
        groups : np.ndarray
            The grouping factors.
        type : str, default="conditional"
            The type of residual to compute. "marginal" uses only fixed effects
            (y - Xb). "conditional" uses both fixed and random effects (y - Xb - Zu).

        Returns
        -------
        np.ndarray of shape (n_samples, n_responses)
            The residuals.
        """
        if type == "marginal":
            return y - self.predict(X)
        elif type == "conditional":
            inference = self.infer(X, y, groups)
            return y - (
                self.predict(X) + inference.observations.total_random_effects.value
            )
        else:
            raise ValueError("type must be 'conditional' or 'marginal'")

    @staticmethod
    def cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """
        Convert a covariance matrix to a correlation matrix.

        Parameters
        ----------
        cov : np.ndarray
            A square covariance matrix.

        Returns
        -------
        np.ndarray
            The corresponding correlation matrix.
        """
        std = np.sqrt(np.maximum(np.diag(cov), 0))
        std[std == 0] = 1e-12
        return cov / np.outer(std, std)

    @property
    def R(self) -> np.ndarray:
        """Get the estimated residual covariance matrix."""
        return self._R.matrix

    @property
    def G(self) -> list[np.ndarray]:
        """Get the estimated covariance matrices for each random effect grouping factor."""
        return [term.matrix for term in self._G]

    @property
    def R_corr(self) -> np.ndarray:
        """The correlation matrix of the residuals."""
        return self.cov_to_corr(self.R)

    @property
    def G_corr(self) -> list[np.ndarray]:
        """A list containing the correlation matrices for the random effects of each grouping factor."""
        return [self.cov_to_corr(term) for term in self.G]

    @property
    def fixed_effects_df(self) -> int:
        """The degrees of freedom (number of coefficients) in the fixed-effects model, if discernible. Otherwise 0."""
        if hasattr(self.fixed_effects_model, "coef_"):
            return self.fixed_effects_model.coef_.size
        return 0

    @property
    def aic(self) -> float:
        """The Akaike Information Criterion (AIC) for model selection."""
        k_params = self.n_responses * (self.n_responses + 1) / 2
        for cov in self._G:
            d = cov.n_effects * self.n_responses
            k_params += d * (d + 1) / 2
        k_params += self.fixed_effects_df
        return 2 * k_params - 2 * self.best_log_likelihood

    @property
    def bic(self) -> float:
        """The Bayesian Information Criterion (BIC) for model selection."""
        k_params = self.n_responses * (self.n_responses + 1) / 2
        for cov in self._G:
            d = cov.n_effects * self.n_responses
            k_params += d * (d + 1) / 2
        k_params += self.fixed_effects_df
        return k_params * np.log(self.n_samples) - 2 * self.best_log_likelihood

    def pseudo_r2(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
        """
        Compute the pseudo R-squared values.

        Calculates both the marginal R-squared (variance explained by fixed effects
        only) and conditional R-squared (variance explained by both fixed and random
        effects).

        Parameters
        ----------
        X : np.ndarray
            The fixed-effects design matrix.
        y : np.ndarray
            The true response matrix.
        groups : np.ndarray
            The grouping factors.

        Returns
        -------
        dict
            A dictionary with keys "marginal" and "conditional" containing the
            respective R-squared values for each output.
        """
        fixed_effects_predictions = self.predict(X)
        inference = self.infer(X, y, groups)

        fixed_effects_var = np.var(fixed_effects_predictions, axis=0)
        re_var = np.var(inference.observations.total_random_effects.value, axis=0)
        res_var = np.diag(self.R)

        total_var = fixed_effects_var + re_var + res_var
        marginal_r2 = fixed_effects_var / total_var
        conditional_r2 = (fixed_effects_var + re_var) / total_var

        return {"marginal": marginal_r2, "conditional": conditional_r2}

    def icc(self) -> dict:
        """
        Compute the Intra-Class Correlation Coefficient (ICC) for each group.

        Returns
        -------
        dict
            A dictionary where keys are group names and values are the proportion
            of total variance explained by the group random intercepts.
        """
        res_var = np.diag(self.R)
        total_re_var = np.zeros(self.n_responses)
        for cov in self._G:
            q = cov.n_effects
            total_re_var += np.array([
                cov.matrix[i * q, i * q] for i in range(self.n_responses)
            ])
        total_var = total_re_var + res_var

        icc_dict = {}
        for k, cov in enumerate(self._G):
            q = cov.n_effects
            intercept_vars = np.array([
                cov.matrix[i * q, i * q] for i in range(self.n_responses)
            ])
            icc_dict[f"Group_{k}"] = intercept_vars / total_var
        return icc_dict

    def blups(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, group_idx: int = 0
    ) -> pd.DataFrame:
        """
        Extract the Best Linear Unbiased Predictors (BLUPs) for a specific grouping factor.

        Parameters
        ----------
        X : np.ndarray
            The fixed-effects design matrix.
        y : np.ndarray
            The true response matrix.
        groups : np.ndarray
            The grouping factors.
        group_idx : int, default=0
            The index of the grouping factor to extract BLUPs for.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the random intercepts and slopes for each unique
            level of the group.
        """
        inference = self.infer(X, y, groups)
        blups_3d = inference.groups[group_idx].effects.value
        blups_val = blups_3d.reshape(blups_3d.shape[0], -1)

        cov = self._G[group_idx]
        q = cov.n_effects
        cols = []
        for i in range(self.n_responses):
            for j in range(q):
                effect_name = "Intercept" if j == 0 else f"Slope {j}"
                cols.append(f"R{i + 1}_{effect_name}")

        unique_levels = inference.groups[group_idx].levels
        df = pd.DataFrame(blups_val, columns=cols, index=unique_levels)
        df.index.name = f"Group_{group_idx}"
        return df

    def marginal_corr(
        self, slope_covariates: list[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Compute the total marginal correlation matrix (m x m) for a single observation profile.

        Parameters
        ----------
        slope_covariates : list of np.ndarray or None, optional
            A list specifying the covariates (excluding the intercept) for each grouping
            factor.

        Returns
        -------
        np.ndarray
            The marginal correlation matrix of shape (n_responses, n_responses).
        """
        return self.cov_to_corr(self.marginal_cov(slope_covariates))

    def marginal_cov(
        self, slope_covariates: list[np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Compute the unconditional marginal covariance matrix for a single new observation.

        Parameters
        ----------
        slope_covariates : list of np.ndarray or None, optional
            A list specifying the covariates (excluding the intercept) for each grouping
            factor.

        Returns
        -------
        np.ndarray
            The marginal covariance matrix of shape (n_responses, n_responses).
        """
        cov_total = self.R.copy()
        if slope_covariates is None:
            slope_covariates = [None] * self.n_groups

        for k, (slopes, cov_k) in enumerate(zip(slope_covariates, self._G)):
            if slopes is None or len(np.atleast_1d(slopes)) == 0:
                z = np.array([1.0])
            else:
                z = np.concatenate(([1.0], np.atleast_1d(slopes)))

            if len(z) != cov_k.n_effects:
                raise ValueError(
                    f"Expected {cov_k.n_effects - 1} slope covariates for group {k + 1}, got {len(z) - 1}."
                )

            m = self.n_responses
            q = cov_k.n_effects
            G_reshaped = cov_k.matrix.reshape((m, q, m, q))
            cov_total += np.einsum("mqnr,q,r->mn", G_reshaped, z, z)

        return cov_total

    def summary(self) -> str:
        """Generate a formatted summary string of the fitted model."""
        lines = []
        indent1 = "  "
        indent2 = "      "

        lines.append("\nMixed Model Summary")
        lines.append("=" * 60)
        lines.append(
            indent1 + f"FE Model:             {type(self.fixed_effects_model).__name__}"
        )
        lines.append(indent1 + f"Iterations:           {len(self.log_likelihood)}")
        status = (
            "Yes (Early Stopped)"
            if self.is_early_stopped
            else ("Yes" if self.is_converged else "No")
        )
        lines.append(indent1 + f"Converged:            {status}")
        lines.append(indent1 + f"Log-Likelihood:       {self.best_log_likelihood:.3f}")

        fe_df_str = (
            str(self.fixed_effects_df)
            if self.fixed_effects_df > 0
            else "0 (Agnostic ML Model)"
        )
        lines.append(indent1 + f"FE DF:                {fe_df_str}")
        lines.append(indent1 + f"AIC:                  {self.aic:.3f}")
        lines.append(indent1 + f"BIC:                  {self.bic:.3f}")
        lines.append(indent1 + f"Outputs (n_responses):{self.n_responses}")
        lines.append(indent1 + f"Groups (n_groups):    {self.n_groups}")

        lines.append("-" * 60)
        lines.append(indent1 + "Residual Variances (R)")
        lines.append(indent2 + "{:<10} {:>12}".format("Response", "Variance"))
        for i in range(self.n_responses):
            lines.append(indent2 + "{:<10} {:>12.4f}".format(i + 1, self.R[i, i]))

        lines.append("-" * 60)
        lines.append(indent1 + "Random Effects Variances (G)")
        lines.append(
            indent2
            + "{:<8} {:<10} {:<15} {:>12}".format(
                "Group", "Response", "Effect", "Variance"
            )
        )

        for k, cov in enumerate(self._G):
            for i in range(self.n_responses):
                for j in range(cov.n_effects):
                    idx = i * cov.n_effects + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    lines.append(
                        indent2
                        + "{:<8} {:<10} {:<15} {:>12.4f}".format(
                            k + 1, i + 1, effect_name, cov.matrix[idx, idx]
                        )
                    )

        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"<MixedModel: {self.n_responses} responses, {self.n_groups} groups (Converged: {self.is_converged})>"
