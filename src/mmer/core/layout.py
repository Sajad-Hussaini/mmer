from __future__ import annotations

import numpy as np


class RandomEffectLayout:
    """
    Exact grouped layout for one random-effect term.

    The layout preserves the current model algebra but stores the design in a
    group-indexed form so matvecs and Gram summaries do not need a sparse matrix.
    Columns follow the existing slope-major ordering:
    [intercept levels, slope 1 levels, slope 2 levels, ...].
    """

    def __init__(self, group: np.ndarray, covariates: np.ndarray | None):
        group = np.asarray(group)
        if group.ndim == 0:
            raise ValueError("Group labels must contain at least one observation.")
        group = group.ravel()

        if covariates is not None:
            covariates = np.asarray(covariates)
            if covariates.ndim == 1:
                covariates = covariates[:, None]
            if covariates.shape[0] != group.shape[0]:
                raise ValueError(
                    f"Covariates row count mismatch. Expected {group.shape[0]}, got {covariates.shape[0]}"
                )

        self.levels, self.level_indices = np.unique(group, return_inverse=True)
        self.o = self.levels.size
        self.covariates = covariates
        self.slope_count = 0 if covariates is None else covariates.shape[1]
        self.q = 1 + self.slope_count
        self.block_size = self.q * self.o

        self._level_gram = self._build_level_gram()
        self._ztz_diag = np.concatenate([self._level_gram[:, slope, slope] for slope in range(self.q)])

    @property
    def ztz_diag(self) -> np.ndarray:
        """Diagonal of Z.T @ Z in the current slope-major ordering."""
        return self._ztz_diag

    def apply(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Apply Z to grouped coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Flattened coefficients with shape (m * q * o,).
        """
        coeffs = np.asarray(coeffs).reshape((-1, self.q, self.o))
        values = coeffs[:, 0, self.level_indices].copy()

        if self.covariates is not None:
            for slope_index, covariate in enumerate(self.covariates.T, start=1):
                values += coeffs[:, slope_index, self.level_indices] * covariate

        return values.ravel()

    def apply_transpose(self, values: np.ndarray) -> np.ndarray:
        """
        Apply Z.T to observation-space values.

        Parameters
        ----------
        values : np.ndarray
            Flattened values with shape (m * n,).
        """
        values = np.asarray(values).reshape((-1, self.level_indices.size))
        result = np.empty((values.shape[0], self.q, self.o), dtype=np.float64)

        for response_index, response in enumerate(values):
            result[response_index, 0] = np.bincount(self.level_indices, weights=response, minlength=self.o)

            if self.covariates is not None:
                for slope_index, covariate in enumerate(self.covariates.T, start=1):
                    result[response_index, slope_index] = np.bincount(
                        self.level_indices,
                        weights=response * covariate,
                        minlength=self.o,
                    )

        return result.ravel()

    def trace_against_block(self, sigma_block: np.ndarray) -> float:
        """
        Compute sum(Z.T @ Z * sigma_block) exactly for one block.
        """
        sigma_block = np.asarray(sigma_block).reshape((self.q, self.o, self.q, self.o))
        total = 0.0
        for left_slope in range(self.q):
            for right_slope in range(self.q):
                total += np.dot(
                    self._level_gram[:, left_slope, right_slope],
                    np.diagonal(sigma_block[left_slope, :, right_slope, :]),
                )
        return float(total)

    def trace_against_block_stack(self, sigma_blocks: np.ndarray) -> np.ndarray | float:
        """
        Compute traces against a stack of coefficient blocks.
        """
        sigma_blocks = np.asarray(sigma_blocks)
        if sigma_blocks.ndim == 2:
            return float(self.trace_against_block(sigma_blocks))
        if sigma_blocks.ndim != 3:
            raise ValueError("Expected a single block matrix or a stack of block matrices.")

        traces = np.empty(sigma_blocks.shape[0], dtype=np.float64)
        for index, sigma_block in enumerate(sigma_blocks):
            traces[index] = self.trace_against_block(sigma_block)
        return traces

    def _build_level_gram(self) -> np.ndarray:
        """
        Build the per-level q x q Gram matrices exactly from the grouped data.
        """
        gram = np.zeros((self.o, self.q, self.q), dtype=np.float64)
        counts = np.bincount(self.level_indices, minlength=self.o).astype(np.float64)
        gram[:, 0, 0] = counts

        if self.covariates is None:
            return gram

        for slope_index, covariate in enumerate(self.covariates.T, start=1):
            sums = np.bincount(self.level_indices, weights=covariate, minlength=self.o).astype(np.float64)
            gram[:, 0, slope_index] = sums
            gram[:, slope_index, 0] = sums

        for left_index in range(self.slope_count):
            left_covariate = self.covariates[:, left_index]
            for right_index in range(left_index, self.slope_count):
                sums = np.bincount(
                    self.level_indices,
                    weights=left_covariate * self.covariates[:, right_index],
                    minlength=self.o,
                ).astype(np.float64)
                left_position = left_index + 1
                right_position = right_index + 1
                gram[:, left_position, right_position] = sums
                if right_index != left_index:
                    gram[:, right_position, left_position] = sums

        return gram