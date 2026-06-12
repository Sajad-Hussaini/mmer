import numpy as np
from dataclasses import dataclass


@dataclass
class Estimate:
    """A generic container for an estimated value and its epistemic uncertainty."""

    value: np.ndarray
    std: np.ndarray | None = None


@dataclass
class ObservationPosterior:
    """Observation-level inference outputs (shape: n_samples x n_responses)"""

    residuals: np.ndarray
    total_random_effects: np.ndarray
    residuals_std: np.ndarray | None = None
    total_random_effects_std: np.ndarray | None = None


@dataclass
class GroupPosterior:
    """Group-level inference outputs (shape: n_levels x n_responses x q)"""

    levels: np.ndarray  # The unique group identifiers
    effects: np.ndarray  # The Best Linear Unbiased Predictors (BLUPs)
    effects_std: np.ndarray | None = None


@dataclass
class InferenceResult:
    """The complete posterior state."""

    observations: ObservationPosterior
    groups: list[GroupPosterior]
