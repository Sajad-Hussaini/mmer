import numpy as np
import copy
from .result import MixedModel
from ..em.algorithm import EMSolver
from ..em.convergence import ConvergenceMonitor
from ..em.corrections import VarianceCorrection
from ..structures.covariance import RandomCovariance, ResidualCovariance
from ..structures.design import RandomDesign


class MixedEffectEstimator:
    """
    Multivariate Mixed-Effects Estimator.

    This estimator structurally wraps a base predictive model (e.g., a neural network,
    random forest, or linear model) to seamlessly learn fixed population-level signals
    while formally isolating random group-level variations and complex covariance matrices.
    It utilizes an Expectation-Maximization (EM) algorithm to iteratively estimate
    variance components.

    Parameters
    ----------
    fixed_effects_model : object
        A predictive base model used to estimate the fixed population effects. This model
        must implement standard `fit(X, y)` and `predict(X)` methods.
    max_iter : int, default=30
        The maximum number of Expectation-Maximization (EM) cycles to perform.
    tol : float, default=1e-6
        The numerical tolerance for declaring convergence. Convergence is reached when the
        relative step change in the log-likelihood drops below this threshold.
    patience : int, default=3
        The number of consecutive iterations to tolerate without significant log-likelihood
        improvement before invoking early stopping.
    correction_method : str, default="bste"
        The trace estimation method for variance components. Options are:
        - `"bste"`: Block-Stochastic Trace Estimator (Highly recommended for large datasets).
        - `"de"`: Deterministic Trace Evaluator (Recommended only for small matrix sizes).
    slq_steps : int, default=30
        The number of Lanczos iterations used for stochastic log-determinant approximations
        in the BSTE method.
    n_probes : int, default=60
        The number of Rademacher random probe vectors used during stochastic trace estimation.
    preconditioner : bool, default=True
        Whether to apply a residual block preconditioner when resolving the linear system,
        significantly reducing Conjugate Gradient (CG) iterations.
    cg_maxiter : int, default=1000
        Maximum allowable Conjugate Gradient solver iterations per step.
    n_jobs : int, default=-1
        Number of parallel threading jobs to run during stochastic trace evaluation.
        Defaults to -1 (use all available processors).
    backend : str, default="threading"
        The parallel execution backend (e.g., `"threading"` or `"loky"`).
    """

    def __init__(
        self,
        fixed_effects_model,
        max_iter: int = 30,
        tol: float = 1e-6,
        patience: int = 3,
        correction_method: str = "bste",
        slq_steps: int = 30,
        n_probes: int = 60,
        preconditioner: bool = True,
        cg_maxiter: int = 1000,
        n_jobs: int = -1,
        backend: str = "threading",
    ):
        self.fixed_effects_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.patience = max(1, patience)
        self.correction_method = correction_method
        self.slq_steps = slq_steps
        self.n_probes = n_probes
        self.preconditioner = preconditioner
        self.cg_maxiter = cg_maxiter
        self.n_jobs = n_jobs
        self.backend = backend

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        random_slopes: tuple = None,
    ) -> "MixedModel":
        """
        Train the mixed-effects framework to separate fixed and random components.

        This function triggers the Expectation-Maximization (EM) engine. It alternates
        between fitting the fixed-effects base model to the partial residuals and updating
        the random covariance matrices using generalized least squares.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The fixed-effects design matrix (model covariates).
        y : ndarray of shape (n_samples, n_responses)
            The multi-output continuous response matrix.
        groups : ndarray of shape (n_samples, n_groups)
            Categorical arrays linking each observation to its respective group levels.
        random_slopes : tuple of tuples, optional
            A tuple of length `n_groups`. Each inner tuple specifies the column indices in `X`
            that should act as random slope covariates for that specific grouping factor.
            `None` denotes a random-intercepts-only model.

        Returns
        -------
        MixedModel
            A fitted structural container retaining the fully converged fixed-effects
            estimator, globally learned variance matrices, and learned group-level 
            effects.
        """
        if y.ndim == 1:
            y = y[:, None]
        if groups.ndim == 1:
            groups = groups[:, None]
        n_samples, n_responses = y.shape
        n_groups = groups.shape[1]

        if random_slopes is None:
            random_slopes_tuple = tuple([None] * n_groups)
        elif isinstance(random_slopes, dict):
            random_slopes_tuple = tuple(
                random_slopes.get(i, None) for i in range(n_groups)
            )
        elif len(random_slopes) != n_groups:
            raise ValueError(
                f"Length of random_slopes ({len(random_slopes)}) must match number of groups ({n_groups})."
            )
        else:
            random_slopes_tuple = tuple(random_slopes)

        random_covs = []
        random_designs = []
        for i, slope_cols in enumerate(random_slopes_tuple):
            n_effects = 1 if slope_cols is None else 1 + len(slope_cols)
            cov = RandomCovariance(n_responses, n_effects)
            covariates = X[:, slope_cols] if slope_cols is not None else None
            d = RandomDesign(groups[:, i], covariates)
            random_covs.append(cov)
            random_designs.append(d)

        resid_cov = ResidualCovariance(n_responses)

        convergence_monitor = ConvergenceMonitor(tol=self.tol, patience=self.patience)
        variance_corrector = VarianceCorrection(
            method=self.correction_method,
            cg_maxiter=self.cg_maxiter,
            n_jobs=self.n_jobs,
            backend=self.backend,
        )

        fixed_effects_instance = copy.deepcopy(self.fixed_effects_model)

        solver = EMSolver(
            fixed_effects_model=fixed_effects_instance,
            max_iter=self.max_iter,
            convergence_monitor=convergence_monitor,
            variance_corrector=variance_corrector,
            preconditioner=self.preconditioner,
            cg_maxiter=self.cg_maxiter,
            n_responses=n_responses,
            n_probes=self.n_probes,
            slq_steps=self.slq_steps,
            n_jobs=self.n_jobs,
            backend=self.backend,
        )

        solver.run(
            X,
            y,
            tuple(random_covs),
            tuple(random_designs),
            resid_cov,
        )

        solver.convergence_monitor.restore_best_state(solver)

        from .result import MixedModel

        model = MixedModel(
            fixed_effects_model=solver.fixed_effects_model,
            n_samples=n_samples,
            n_responses=n_responses,
            n_groups=n_groups,
            G=solver.random_covs,
            R=solver.resid_cov,
            log_likelihood=solver.convergence_monitor.log_likelihood,
            is_converged=solver.convergence_monitor.is_converged,
            is_early_stopped=solver.convergence_monitor.is_early_stopped,
            best_log_likelihood=solver.convergence_monitor._best_log_likelihood,
            preconditioner=self.preconditioner,
            cg_maxiter=self.cg_maxiter,
            force_iterative=solver.force_iterative,
            random_slopes=random_slopes_tuple,
        )

        train_inference = model.infer(X, y, groups)
        model.training_group_posteriors = train_inference.groups

        return model
