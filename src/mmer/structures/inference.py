import numpy as np
from ..linalg.solver import build_solver
from .covariance import RandomCovariance, ResidualCovariance
from .design import RandomDesign
from ..linalg.operator import KroneckerMath


def aggregate_random_effects(
    preconditioned_residuals: np.ndarray,
    random_covs: tuple[RandomCovariance, ...],
    random_designs: tuple[RandomDesign, ...],
    n: int,
) -> tuple:
    """
    Compute the Best Linear Unbiased Predictors (BLUPs) and their aggregated sum.

    Mathematically, this calculates the individual random effect components
    :math:`\\mu_k = D_k (I_m \\otimes Z_k^T) P \\epsilon`, and maps them back into
    the observation space to form the total random effect.

    Parameters
    ----------
    preconditioned_residuals : np.ndarray
        The preconditioned residuals :math:`P \\epsilon` of shape (n_responses * n_samples,).
    random_covs : tuple of RandomCovariance
        The list of random covariance state containers.
    random_designs : tuple of RandomDesign
        The corresponding design structures (sparse Z matrices) for each grouping factor.
    n : int
        The number of samples.

    Returns
    -------
    tuple
        - total_random_effect (np.ndarray): The sum of all random effects in observation space.
        - mu (tuple of np.ndarray): The individual BLUPs in their respective group spaces.
    """
    total_random_effect = np.zeros_like(preconditioned_residuals)
    mu = []

    for cov, d in zip(random_covs, random_designs):
        # Compute mu = D (I_m \otimes Z^T) @ preconditioned_residuals
        val = KroneckerMath._kronZ_D_T_matvec(
            d.Z,
            cov.matrix,
            preconditioned_residuals,
            cov.n_responses,
            cov.n_effects,
            d.n_levels,
            n,
        )
        mu.append(val)

        # map mu to total = (I_m \otimes Z) @ mu
        total_random_effect += KroneckerMath._kronZ_matvec(
            d.Z, val, cov.n_responses, cov.n_effects, d.n_levels, n
        )

    return total_random_effect, tuple(mu)


def compute_random_effects_posterior(
    random_covs: tuple[RandomCovariance, ...],
    random_designs: tuple[RandomDesign, ...],
    resid_cov: ResidualCovariance,
    y: np.ndarray,
    fe_predictions: np.ndarray,
    preconditioner: bool = True,
    cg_maxiter: int = 1000,
    force_iterative: bool = False,
) -> tuple:
    """
    Solve for the Best Linear Unbiased Predictors (BLUPs) and conditional residuals.

    This function isolates the inference logic. It builds the system solver for
    the marginal covariance matrix :math:`V`, computes the preconditioned residuals,
    and then extracts the specific random effect posteriors for each group.

    Parameters
    ----------
    random_covs : tuple of RandomCovariance
        The list of random covariance state containers.
    random_designs : tuple of RandomDesign
        The corresponding sparse design structures.
    resid_cov : ResidualCovariance
        The residual covariance state container.
    y : np.ndarray
        The multi-output response matrix of shape (n, self.n_responses).
    fe_predictions : np.ndarray
        The fixed-effects predictions of shape (n, self.n_responses).
    preconditioner : bool, default=True
        Whether to apply a residual block preconditioner for the iterative solver.
    cg_maxiter : int, default=1000
        Maximum iterations for the Conjugate Gradient solver if the iterative path is taken.
    force_iterative : bool, default=False
        If True, prevents falling back to the exact Woodbury solver.

    Returns
    -------
    tuple
        - residuals_2d (np.ndarray): The conditional residuals of shape (n, self.n_responses).
        - total_effect_2d (np.ndarray): The combined random effects mapped to observation space.
        - mu_reshaped (tuple of np.ndarray): The BLUPs reshaped to (groups, self.n_responses, q) for each factor.
    """
    n_samples, n_responses = y.shape
    result_dtype = np.promote_types(y.dtype, fe_predictions.dtype)
    if not np.issubdtype(result_dtype, np.floating):
        result_dtype = np.float64
    marginal_residuals_2d = np.empty((n_responses, n_samples), dtype=result_dtype)
    np.subtract(y.T, fe_predictions.T, out=marginal_residuals_2d)
    marginal_residuals = marginal_residuals_2d.ravel()

    solver = build_solver(
        random_covs,
        random_designs,
        resid_cov,
        n_samples,
        preconditioner,
        cg_maxiter,
        force_iterative,
    )
    preconditioned_residuals = solver.solve(marginal_residuals)

    total_random_effect, mu = aggregate_random_effects(
        preconditioned_residuals, random_covs, random_designs, n_samples
    )

    residuals = marginal_residuals - total_random_effect

    residuals_2d = residuals.reshape((n_responses, -1)).T
    total_effect_2d = total_random_effect.reshape((n_responses, -1)).T

    mu_reshaped = []
    for k, mu_k in enumerate(mu):
        n_effects = random_covs[k].n_effects
        mu_reshaped.append(
            mu_k.reshape((n_responses, n_effects, -1)).transpose(2, 0, 1)
        )

    return residuals_2d, total_effect_2d, tuple(mu_reshaped)
