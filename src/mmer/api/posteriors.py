import numpy as np
from typing import NamedTuple


class Estimate(NamedTuple):
    """
    A unified structural container holding expected numerical estimates and their epistemic uncertainties.

    In the context of the `mmer` package, parameters such as variance matrices (``G``, ``R``)
    or specific random effects (BLUPs) are represented using this structure. For standard
    single-model fits, the standard deviation (`std`) is mathematically omitted (`None`).
    When utilizing the `EnsembleMixedModel`, the `std` array quantifies the epistemic
    uncertainty (disagreement) across the base estimators.

    Attributes
    ----------
    value : ndarray
        The expected point value (e.g., the mean estimate across an ensemble, or the exact
        estimate of a single mixed model).
    std : ndarray or None, optional
        The sample standard deviation representing epistemic uncertainty. For a single model,
        this is intrinsically `None`.
    """

    value: np.ndarray
    std: np.ndarray | None = None


class ObservationPosterior(NamedTuple):
    """
    Observation-level inference outputs mathematically reconstructed for a specific dataset.

    This structure is returned exclusively during inference (i.e., when calling `infer()` with
    a known outcome array `y`). It contains exact point-wise evaluations of errors and group
    contributions.

    Attributes
    ----------
    residuals : Estimate
        The conditional residuals mathematically defined as ``e = y - X * beta - Z * u``. These represent
        the remaining unexplained variance for each specific observation after accounting for both
        fixed population trends and random group offsets.
    total_random_effects : Estimate
        The aggregated sum of all random effects mapped back into observation space (``Z * u``). This
        is the exact amount by which group properties shifted the prediction for each sample.

    Notes
    -----
    These quantities are strictly tied to the dataset provided during inference and cannot be
    directly reused for new predictions. New test observations must calculate their
    own random effects using `MixedModel.predict()`.
    """

    residuals: Estimate
    total_random_effects: Estimate


class GroupPosterior(NamedTuple):
    """
    Group-level inference outputs representing the learned Best Linear Unbiased Predictors (BLUPs).

    These structures form the absolute core of personalized prediction. During training, the EM
    algorithm solves for the latent group properties (BLUPs). These properties are learned
    and stored so they can be seamlessly applied to future predictions for known group levels.

    Attributes
    ----------
    levels : ndarray
        A 1D array of the unique categorical identifiers defining the group levels.
    counts : ndarray
        A 1D array describing the number of instances or observations assigned to each group level.
    effects : Estimate
        The Best Linear Unbiased Predictors (BLUPs) forming the random effect matrix $u$.
        The value is an `ndarray` of shape `(n_levels, n_responses, n_effects)`, detailing the exact
        multivariate offset and slope adjustments for every specific group level.
    """

    levels: np.ndarray
    counts: np.ndarray
    effects: Estimate


class InferenceResult(NamedTuple):
    """
    The complete analytical posterior state returned by a mixed-effects inference.

    This container cleanly encapsulates both the observation-level metrics (e.g., specific
    sample residuals) and the group-level metrics (e.g., learned BLUP offsets). It is the
    primary return structure of the `infer()` method.

    Attributes
    ----------
    observations : ObservationPosterior
        The sample-specific properties resulting from the exact mathematical breakdown of
        ``y = X * beta + Z * u + e``.
    groups : list of GroupPosterior
        A discrete list containing the group-level offsets (BLUPs) and metadata for each
        grouping factor included in the model formulation.
    """

    observations: ObservationPosterior
    groups: list[GroupPosterior]
