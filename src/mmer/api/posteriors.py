import numpy as np
from typing import NamedTuple


class Estimate(NamedTuple):
    """
    A generic container for an estimated value and its epistemic uncertainty.

    Attributes
    ----------
    value : np.ndarray
        The estimated point value.
    std : np.ndarray, optional
        The epistemic standard deviation. None for standard mixed models.
    """

    value: np.ndarray
    std: np.ndarray | None = None


class ObservationPosterior(NamedTuple):
    """
    Observation-level inference outputs.

    Attributes
    ----------
    residuals : Estimate
        The estimated residuals (y - Xb - Zu).
    total_random_effects : Estimate
        The total random effects summed across all grouping factors (Zu).
    """

    residuals: Estimate
    total_random_effects: Estimate


class GroupPosterior(NamedTuple):
    """
    Group-level inference outputs.

    Attributes
    ----------
    levels : np.ndarray
        The unique group identifiers.
    counts : np.ndarray
        The number of observations for each level.
    effects : Estimate
        The Best Linear Unbiased Predictors (BLUPs) for each level.
    """

    levels: np.ndarray
    counts: np.ndarray
    effects: Estimate


class InferenceResult(NamedTuple):
    """
    The complete posterior state.

    Attributes
    ----------
    observations : ObservationPosterior
        The observation-level results (residuals and total random effects).
    groups : list of GroupPosterior
        A list containing the group-level results for each grouping factor.
    """

    observations: ObservationPosterior
    groups: list[GroupPosterior]
