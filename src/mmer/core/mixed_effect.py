import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from .operator import VLinearOperator, ResidualPreconditioner
from .corrections import VarianceCorrection
from .solver import build_solver
from .convergence import ConvergenceMonitor
from .inference import aggregate_random_effects, compute_random_effects_posterior
from ..lanczos_algorithm import slq
from .terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual


class MixedEffectRegressor:
    """
    Multivariate Mixed Effects Regression (MMER) using Expectation-Maximization.

    Fits mixed model with multiple responses, supporting arbitrary grouping factors 
    and linear random slopes. Solves for random effects and residual covariances
    using EM algorithm with stochastic log-determinant estimation.

    Parameters
    ----------
    fixed_effects_model : RegressorMixin
        Base regressor for fixed effects (must support multi-output).
    max_iter : int, default=20
        Maximum number of EM iterations.
    tol : float, default=1e-6
        Convergence tolerance on log-likelihood relative change.
    patience : int, default=3
        Number of iterations to wait for likelihood improvement before early stopping.
        Setting to a high value effectively disables early stopping and relies solely on `tol`.
    slq_steps : int, default=30
        Number of Lanczos steps for Stochastic Lanczos Quadrature (log-det estimation).
        A range of 30-50 is typically sufficient. Higher values yield slightly more accurate estimates but increase computation time and risk numerical instability.
    n_probes : int, default=60
        Number of random probes used for both SLQ log-determinant estimation and stochastic variance correction.
        A fixed target around 50-60 is usually optimal independent of matrix dimension for O(1/sqrt(p)) error convergence.
    preconditioner : bool, default=True
        Whether to use residual-based preconditioner for CG solver.
    correction_method : str, default='bste'
        Method for variance correction in M-step:
        
        - 'ste': stochastic trace estimation
        - 'bste': block stochastic trace estimation
        - 'de': deterministic estimation
    n_jobs : int, default=-1
        Number of parallel jobs for SLQ and trace estimation (-1 uses all cores).
        Setting to number of outputs (`m`) is recommended for optimal performance.
    backend : str, default='loky'
        Joblib parallel backend ('loky', 'threading').
        Setting to 'loky' is recommended for CPU-bound tasks.
    
    Attributes
    ----------
    fe_model : RegressorMixin
        Fitted fixed effects model.
    random_effect_terms : tuple of RandomEffectTerm
        Fitted random effect terms containing covariance matrices.
    residual_term : ResidualTerm
        Fitted residual term containing residual covariance matrix.
    log_likelihood : list of float
        Log-likelihood values across EM iterations.
    n : int
        Number of observations.
    m : int
        Number of output dimensions.
    k : int
        Number of grouping factors.
    
    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> model = MixedEffectRegressor(fixed_effects_model=Ridge())
    >>> results = model.fit(X, y, groups, random_slopes=([0, 1], None))
    >>> predictions = model.predict(X_new)
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 30, tol: float = 1e-6, patience: int = 3,
                 slq_steps: int = 30, n_probes: int = 60, preconditioner: bool = True, correction_method: str = 'bste',
                 cg_maxiter: int = 1000, n_jobs: int = -1, backend: str = 'loky'):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.patience = max(1, patience)
        self.slq_steps = slq_steps
        self.n_probes = n_probes
        self.preconditioner = preconditioner
        self.correction_method = correction_method
        self.cg_maxiter = cg_maxiter
        self.n_jobs = n_jobs
        self.backend = backend

        self.convergence_monitor = ConvergenceMonitor(tol=tol, patience=patience)
        self.variance_corrector = VarianceCorrection(method=correction_method, cg_maxiter=cg_maxiter, n_jobs=n_jobs, backend=backend)
        
        # State: Terms
        self.random_effect_terms: tuple[RandomEffectTerm] = None # List[RandomEffectTerm]
        self.residual_term: ResidualTerm = None # ResidualTerm
    
    @property
    def log_likelihood(self):
        """
        Log-likelihood history from convergence monitor.
        
        Returns
        -------
        list of float
            Log-likelihood values for each EM iteration.
        """
        return self.convergence_monitor.log_likelihood
    
    @property
    def _is_converged(self):
        """
        Convergence status from convergence monitor.
        
        Returns
        -------
        bool
            Whether the model has converged.
        """
        return self.convergence_monitor.is_converged
    
    @property
    def _best_log_likelihood(self):
        """
        Best log-likelihood value encountered during fitting.
        
        Returns
        -------
        float
            Maximum log-likelihood achieved.
        """
        return self.convergence_monitor._best_log_likelihood

    def _prepare_terms(self, y: np.ndarray, groups: np.ndarray, random_slopes: tuple[list[int] | None] | None):
        """
        Initialize state RandomEffect and Residual Terms if not present.
        """
        self.n, self.m = y.shape  # number of sample and outputs
        self.k = groups.shape[1]  # number of groups
        
        # 1. Initialize Random Structure Config
        if random_slopes is None:
            config_random_slopes = tuple([None] * self.k)
        elif len(random_slopes) != self.k:
             raise ValueError(f"Length of random_slopes ({len(random_slopes)}) must match number of groups ({self.k}).")
        else:
            config_random_slopes = random_slopes
            
        # 2. Create Terms
        self.random_effect_terms = []
        for i, slope_cols in enumerate(config_random_slopes):
            term = RandomEffectTerm(group_id=i, covariates_id=slope_cols, m=self.m)
            self.random_effect_terms.append(term)
            
        self.residual_term = ResidualTerm(m=self.m)
        self.random_slopes = config_random_slopes

    def _realize_objects(self, X: np.ndarray, groups: np.ndarray) -> tuple:
        """
        Factory method to create realized random effects and residual term.
        
        Parameters
        ----------
        X : np.ndarray
            Covariates, shape (n, p).
        y : np.ndarray
            Multi-output targets, shape (n, m).
        groups : np.ndarray
            Grouping factors, shape (n, k).
        
        Returns
        -------
        realized_effects : tuple of RealizedRandomEffect
            Realized random effects.
        realized_residual : RealizedResidual
            Realized residual term.
        """
        n = X.shape[0]
        realized_effects = tuple(RealizedRandomEffect(term, X, groups) for term in self.random_effect_terms)
        realized_residual = RealizedResidual(self.residual_term, n)
        return realized_effects, realized_residual

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                     validation_split: float = 0.0, validation_group: int = 0):
        """
        Prepare data for EM algorithm by creating realized objects.
        
        Generates transient realized random effects and residual for the current 
        dataset. Optionally splits data into training and validation sets based on
        group membership.
        
        Parameters
        ----------
        X : np.ndarray
            Covariates, shape (n, p).
        y : np.ndarray
            Multi-output targets, shape (n, m).
        groups : np.ndarray
            Grouping factors, shape (n, k).
        validation_split : float, default=0.0
            Fraction of groups to use for validation (0.0 means no validation).
            Setting to a non-zero value means fixed effects can accept validation data.
        validation_group : int, default=0
            Column index in `groups` to use for group-wise validation splitting.
        
        Returns
        -------
        marginal_residual : np.ndarray
            Initial marginal residual, raveled shape (m*n,).
        realized_effects : tuple of RealizedRandomEffect
            Realized random effect objects.
        realized_residual : RealizedResidual
            Realized residual term.
        """        
        # Setup Validation Split
        if validation_split > 0:
            main_group = groups[:, validation_group]
            gss = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
            self.train_idx, self.val_idx = next(gss.split(X, y, groups=main_group))
            self.has_validation = True
        else:
            self.train_idx = np.arange(self.n)
            self.val_idx = None
            self.has_validation = False
            
        # Instantiate Realized Objects (Transient)
        realized_effects, realized_residual = self._realize_objects(X, groups)
        
        # Initial Marginal Residual
        marginal_residual = self._compute_marginal_residual(X, y, 0.0)
        
        return marginal_residual, realized_effects, realized_residual

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | tuple[list[int]] = None,
            validation_split: float = 0.0, validation_group: int = 0):
        """
        Fit the MMER model using the EM algorithm.

        Parameters
        ----------
        X : np.ndarray
            Covariates, shape (n, p) where n is number of observations and p is
            number of features.
        y : np.ndarray
            Multi-output targets, shape (n, m) where m is number of outputs.
        groups : np.ndarray
            Grouping factors, shape (n, k) where k is number of grouping factors.
            Each column represents a different grouping structure.
        random_slopes : tuple of list of int, optional
            Tuple of lists specifying random slopes for each grouping factor.
            Each list contains column indices in X for random slopes corresponding 
            to that group. None or empty list implies random intercept only for
            that group. If None, all groups get random intercepts only.
        validation_split : float, default=0.0
            Fraction of groups to use for validation (early stopping). Must be
            between 0.0 and 1.0. Set to 0.0 to disable validation.
            Setting to a non-zero value means fixed effects can accept validation data.
        validation_group : int, default=0
            Column index in `groups` to use for group-wise validation splitting.

        Returns
        -------
        MixedEffectResults
            Fitted result object containing covariance estimates and diagnostics.
        
        Examples
        --------
        >>> # Fit model with random intercepts only
        >>> results = model.fit(X, y, groups)
        
        >>> # Fit with random slopes on features 0 and 1 for first group
        >>> results = model.fit(X, y, groups, random_slopes=([0, 1], None))
        
        >>> # Fit with validation split
        >>> results = model.fit(X, y, groups, validation_split=0.2)
        """
        # Initialize terms if new training
        if self.random_effect_terms is None:
            self._prepare_terms(y, groups, random_slopes)
        
        # Reset convergence monitor for new fit
        self.convergence_monitor.reset()
            
        marginal_residual, realized_effects, realized_residual = self.prepare_data(X, y, groups, validation_split, validation_group)
        
        pbar = tqdm(range(1, self.max_iter + 1), desc="Running MMER | Model Fitting ...", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        
        for _ in pbar:
            marginal_residual = self._run_em_iteration(X, y, marginal_residual, realized_effects, realized_residual)
            if self.convergence_monitor.is_converged:
                pbar.set_description(f"Model Converged | Early stopping ...")
                break
                
        from .mixed_result import MixedEffectResults
        return MixedEffectResults(self)

    def _run_em_iteration(self, X, y, marginal_residual, realized_effects, realized_residual):
        """
        Run one EM iteration.
        """
        total_random_effect, mu, V_op, M_op = self._e_step(marginal_residual, realized_effects, realized_residual)
        
        if self.convergence_monitor.is_converged:
            return marginal_residual
            
        marginal_residual = self._compute_marginal_residual(X, y, total_random_effect.reshape((self.m, self.n)).T)
        self._m_step(marginal_residual, total_random_effect, mu, realized_effects, realized_residual, V_op, M_op)
        
        return marginal_residual

    def _e_step(self, marginal_residual, realized_effects, realized_residual):
        """
        Run E-step.
        """
        solver = build_solver(realized_effects, realized_residual, self.preconditioner, self.cg_maxiter)
        prec_resid, V_op, M_op = solver.solve(marginal_residual)
        
        current_log_lh = self._compute_log_likelihood(marginal_residual, prec_resid, V_op)
        
        # Update convergence monitor
        current_state = {
            're_covs': [term.cov.copy() for term in self.random_effect_terms],
            'resid_cov': self.residual_term.cov.copy(),
            'fe_model': self.fe_model
        }
        self.convergence_monitor.update(current_log_lh, current_state)
        
        if self.convergence_monitor.is_converged:
             return None, None, None, None
             
        total_random_effect, mu = aggregate_random_effects(prec_resid, realized_effects)
        return total_random_effect, mu, V_op, M_op

    def _m_step(self, marginal_residual: np.ndarray, total_random_effect: np.ndarray, mu: tuple[np.ndarray],
                realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual, V_op: VLinearOperator, M_op: ResidualPreconditioner):
        """
        Run M-step.
        """
        eps = marginal_residual - total_random_effect
        T_sum = np.zeros((self.m, self.m))
        new_covs = []
        
        for k, re in enumerate(realized_effects):
            T_k, W_k = self.variance_corrector.compute_correction(k, V_op, M_op, n_probes=self.n_probes)
            T_sum += T_k
            new_covs.append(re._compute_next_cov(mu[k], W_k))

        # Update Terms via Realized Effects logic
        new_resid_cov = realized_residual._compute_next_cov(eps, T_sum)
        
        self.residual_term.set_cov(new_resid_cov)
        for k, new_cov in enumerate(new_covs):
            self.random_effect_terms[k].set_cov(new_cov)

        return self

    def _compute_marginal_residual(self, X, y, total_random_effect):
        """
        Fit FE model and compute new marginal residual.
        """
        y_adj = y - total_random_effect
        y_adj = y_adj if self.m != 1 else y_adj.ravel()

        if self.has_validation:
            X_train = X[self.train_idx]
            y_adj_train = y_adj[self.train_idx]
            X_val = X[self.val_idx]
            y_adj_val = y_adj[self.val_idx]
            self.fe_model.fit(X_train, y_adj_train, X_val=X_val, y_val=y_adj_val)
        else:
            self.fe_model.fit(X, y_adj)

        fx = self.fe_model.predict(X)
        fx = fx if self.m != 1 else fx[:, None]

        return (y - fx).T.ravel()

    def _compute_log_likelihood(self, marginal_residual, prec_resid, V_op):
        """
        Compute log-likelihood.
        """
        log_det_V = slq.logdet(V_op, self.slq_steps, self.n_probes)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + marginal_residual.T @ prec_resid) / 2
        return log_likelihood
