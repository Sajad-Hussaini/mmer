import pickle
import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import solve
from sklearn.base import RegressorMixin
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from .operator import VLinearOperator, ResidualPreconditioner, compute_cov_correction
from ..lanczos_algorithm import slq
from .terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual


# ====================== Helper Classes ======================

class SolverContext:
    """
    Encapsulates solver setup and execution.
    
    Handles preconditioner creation, VLinearOperator setup, and CG invocation
    in one place to eliminate duplicated solver code.
    
    Parameters
    ----------
    realized_effects : tuple of RealizedRandomEffect
        Realized random effects.
    realized_residual : RealizedResidual
        Realized residual term (for preconditioner).
    preconditioner : bool, default=True
        Whether to use preconditioner.
    
    Attributes
    ----------
    realized_effects : tuple of RealizedRandomEffect
        Stored realized random effects.
    realized_residual : RealizedResidual
        Stored realized residual term.
    n : int
        Dataset size, extracted from realized_residual.
    m : int
        Number of outputs, extracted from realized_residual.
    use_preconditioner : bool
        Whether preconditioner will be applied.
    """
    def __init__(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual, preconditioner: bool = True):
        self.realized_effects = realized_effects
        self.realized_residual = realized_residual
        self.n = realized_residual.n
        self.m = realized_residual.m
        self.use_preconditioner = preconditioner
    
    def solve(self, marginal_residual: np.ndarray) -> tuple:
        """
        Solve V * x = marginal_residual using conjugate gradient.
        
        Parameters
        ----------
        marginal_residual : np.ndarray
            Right-hand side of linear system, shape (m*n,).
        
        Returns
        -------
        prec_resid : np.ndarray
            Solution vector, shape (m*n,).
        V_op : VLinearOperator
            Linear operator used in solve.
        M_op : ResidualPreconditioner or None
            Preconditioner used (if any).
        """
        V_op = VLinearOperator(self.realized_effects, self.realized_residual)
        M_op = None
        
        if self.use_preconditioner:
            try:
                resid_cov_inv = solve(a=self.realized_residual.cov, b=np.eye(self.m), assume_a='pos')
                M_op = ResidualPreconditioner(resid_cov_inv, self.n, self.m)
            except Exception:
                pass
        
        prec_resid, info = cg(A=V_op, b=marginal_residual, M=M_op)
        if info != 0:
            print(f"Warning: CG info={info}")
        
        return prec_resid, V_op, M_op


class ConvergenceMonitor:
    """
    Tracks convergence state during EM iterations.
    
    Manages log-likelihood history, patience counter, and best-state restoration.
    Supports both relative tolerance-based stopping and early stopping based on
    patience when no improvement is observed.
    
    Parameters
    ----------
    tol : float, default=1e-6
        Convergence tolerance on log-likelihood relative change.
    patience : int, default=3
        Number of iterations to wait before early stopping if no improvement.
    
    Attributes
    ----------
    tol : float
        Convergence tolerance.
    patience : int
        Patience counter threshold.
    log_likelihood : list of float
        History of log-likelihood values across iterations.
    is_converged : bool
        Whether convergence criteria have been met.
    """
    def __init__(self, tol: float = 1e-6, patience: int = 3):
        self.tol = tol
        self.patience = max(1, patience)
        self.log_likelihood = []
        self.is_converged = False
        self._best_log_likelihood = -np.inf
        self._no_improvement_count = 0
        self._best_state = None
    
    def update(self, current_log_likelihood: float, current_state: dict) -> bool:
        """
        Update convergence monitor with new log-likelihood value.
        
        Checks both relative change tolerance and patience-based early stopping.
        Stores the best state encountered during optimization.
        
        Parameters
        ----------
        current_log_likelihood : float
            Current iteration's log-likelihood value.
        current_state : dict
            Current model state containing 're_covs', 'resid_cov', and 'fe_model'
            to save if this is the best state so far.
        
        Returns
        -------
        is_converged : bool
            Whether model has converged based on tolerance or patience.
        """
        self.log_likelihood.append(current_log_likelihood)
        
        # Check relative change convergence
        if len(self.log_likelihood) >= 2:
            change = np.abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2])
            if change <= self.tol:
                self.is_converged = True
        
        # Track best state
        if current_log_likelihood > self._best_log_likelihood:
            self._best_log_likelihood = current_log_likelihood
            self._no_improvement_count = 0
            self._best_state = {k: v.copy() if isinstance(v, np.ndarray) else pickle.loads(pickle.dumps(v)) 
                               for k, v in current_state.items()}
        else:
            self._no_improvement_count += 1
        
        # Check patience stopping
        if self._no_improvement_count >= self.patience:
            self.is_converged = True
        
        return self.is_converged


class InferenceEngine:
    """
    Handles post-fit inference computations for random effects and residuals.
    
    Decouples inference logic from the EM fitting loop, allowing reuse across
    compute_random_effects() and enhanced predict() methods. Computes posterior
    means of random effects given fitted model parameters.
    
    Parameters
    ----------
    random_effect_terms : tuple of RandomEffectTerm
        Learned random effect terms (fitted state).
    residual_term : ResidualTerm
        Learned residual term (fitted state).
    n : int
        Dataset size.
    preconditioner : bool, default=True
        Whether to use preconditioner in solver.
    
    Attributes
    ----------
    random_effect_terms : tuple of RandomEffectTerm
        Stored random effect terms.
    residual_term : ResidualTerm
        Stored residual term.
    n : int
        Dataset size.
    m : int
        Number of outputs, extracted from residual_term.
    preconditioner : bool
        Whether to use preconditioner.
    """
    def __init__(self, random_effect_terms: tuple[RandomEffectTerm], residual_term: ResidualTerm, n: int, preconditioner: bool = True):
        self.random_effect_terms = random_effect_terms
        self.residual_term = residual_term
        self.n = n
        self.m = residual_term.m
        self.preconditioner = preconditioner
    
    def compute_random_effects(self, realized_effects: tuple[RealizedRandomEffect], realized_residual: RealizedResidual,
                               y: np.ndarray, fe_predictions: np.ndarray) -> tuple:
        """
        Compute posterior mean random effects and residuals.
        
        Given observations and fixed effect predictions, computes the posterior
        mean of random effects for each grouping factor and the final residuals.
        
        Parameters
        ----------
        realized_effects : tuple of RealizedRandomEffect
            Realized random effect objects for current data.
        realized_residual : RealizedResidual
            Realized residual term for current data.
        y : np.ndarray
            Target values, shape (n, m).
        fe_predictions : np.ndarray
            Fixed effect predictions, shape (n, m).
        
        Returns
        -------
        residuals : np.ndarray
            Final residuals after subtracting all effects, raveled shape (m*n,).
        random_effects_sum : np.ndarray
            Sum of random effects across all terms, raveled shape (m*n,).
        mu : tuple of np.ndarray
            Posterior means for each random effect term.
        """
        # Compute marginal residual (before random effects)
        marginal_resid = (y - fe_predictions).T.ravel()
        
        # Solve for random effects
        solver_ctx = SolverContext(realized_effects, realized_residual, self.preconditioner)
        prec_resid, _, _ = solver_ctx.solve(marginal_resid)
        
        # Aggregate random effects
        total_random_effect = np.zeros(self.m * self.n)
        mu = []
        for re in realized_effects:
            val = re._compute_mu(prec_resid)
            mu.append(val)
            total_random_effect += re._map_mu(val)
        
        # Compute final residuals (after subtracting random effects)
        residuals = marginal_resid - total_random_effect
        
        return residuals, total_random_effect, tuple(mu)


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
    slq_steps : int, default=50
        Number of Lanczos steps for Stochastic Lanczos Quadrature (log-det estimation).
        A range of 30-100 is typically sufficient. Higher values yield more accurate estimates but increase computation time.
    slq_probes : int, default=50
        Number of random probes for SLQ log-determinant estimation.
        A range of 30-100 is typically sufficient. Higher values yield more accurate estimates but increase computation time.
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
    _VALID_CORRECTION_METHODS = ['ste', 'bste', 'de']

    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 20, tol: float = 1e-6, patience: int = 3,
                 slq_steps: int = 50, slq_probes: int = 50, preconditioner: bool = True, correction_method: str = 'bste',
                 n_jobs: int = -1, backend: str = 'loky'):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.patience = max(1, patience)
        self.slq_steps = slq_steps
        self.slq_probes = slq_probes
        self.preconditioner = preconditioner
        self.correction_method = correction_method
        self.n_jobs = n_jobs
        self.backend = backend

        self.convergence_monitor = ConvergenceMonitor(tol=tol, patience=patience)
        
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

    def _realize_objects(self, n: int, X: np.ndarray, groups: np.ndarray) -> tuple:
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
        realized_effects, realized_residual = self._realize_objects(self.n, X, groups)
        
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
        self.convergence_monitor = ConvergenceMonitor(tol=self.tol, patience=self.patience)
            
        marginal_residual, realized_effects, realized_residual = self.prepare_data(X, y, groups, validation_split, validation_group)
        
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        
        for _ in pbar:
            marginal_residual = self._run_em_iteration(X, y, marginal_residual, realized_effects, realized_residual)
            if self.convergence_monitor.is_converged:
                pbar.set_description(f"Model Converged | Early stopping.")
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
        solver_ctx = SolverContext(realized_effects, realized_residual, self.preconditioner)
        prec_resid, V_op, M_op = solver_ctx.solve(marginal_residual)
        
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
             
        total_random_effect, mu = self._aggregate_random_effects(prec_resid, realized_effects)
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
            T_k, W_k = compute_cov_correction(k, V_op, M_op, self.correction_method, self.n_jobs, self.backend)
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
        log_det_V = slq.logdet(V_op, self.slq_steps, self.slq_probes, self.n_jobs, self.backend)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + marginal_residual.T @ prec_resid) / 2
        return log_likelihood

    def _aggregate_random_effects(self, prec_resid: np.ndarray, realized_effects: tuple[RealizedRandomEffect]):
        """
        Aggregate random effects.
        """
        total_random_effect = np.zeros(self.m * self.n)
        mu = []
        for re in realized_effects:
            mu.append(re._compute_mu(prec_resid))
            total_random_effect += re._map_mu(mu[-1])
        return total_random_effect, tuple(mu)

    # ================= Public Inference Methods =================
    
    def compute_random_effects(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Compute posterior mean of random effects for new data.
        
        Given a fitted model, computes the posterior distribution of random effects
        and residuals for new observations. Useful for prediction and model diagnostics.
        
        Parameters
        ----------
        X : np.ndarray
            Covariates for new data, shape (n_new, p).
        y : np.ndarray
            Observed targets for new data, shape (n_new, m).
        groups : np.ndarray
            Grouping factors for new data, shape (n_new, k).
        
        Returns
        -------
        residuals : np.ndarray
            Final residuals after subtracting all effects, raveled shape (m*n_new,).
        total_effect : np.ndarray
            Sum of random effects across all terms, raveled shape (m*n_new,).
        mu : tuple of np.ndarray
            Posterior means for each random effect term. Each element corresponds
            to one grouping factor.
        
        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        
        Examples
        --------
        >>> residuals, total_re, mu = model.compute_random_effects(X_new, y_new, groups_new)
        >>> # mu[0] contains random effects for first grouping factor
        >>> # mu[1] contains random effects for second grouping factor (if present)
        """
        if self.random_effect_terms is None:
            raise RuntimeError("Model is not fitted.")
            
        n, m = y.shape
        realized_effects, realized_residual = self._realize_objects(n, X, groups)
        
        # Predict Fixed Effects
        fx = self.fe_model.predict(X)
        fx = fx if self.m != 1 else fx[:, None]
        
        # Use InferenceEngine to compute random effects
        inference_engine = InferenceEngine(self.random_effect_terms, self.residual_term, n, self.preconditioner)
        residuals, total_effect, mu = inference_engine.compute_random_effects(realized_effects, realized_residual, y, fx)
        residuals_2d = residuals.reshape((self.m, -1)).T  # shape (n, m)
        total_effect_2d = total_effect.reshape((self.m, -1)).T  # shape (n, m)

        # Reshape mu for each random effect term: (m*q*o) -> (o, m, q)
        mu_reshaped = []
        for k, mu_k in enumerate(mu):
            m, q = self.m, self.random_effect_terms[k].q
            mu_reshaped.append(mu_k.reshape((m, q, -1)).transpose(2, 0, 1))  # (o, m, q)

        return residuals_2d, total_effect_2d, tuple(mu_reshaped)

    def predict(self, X: np.ndarray):
        """
        Predict using fixed effects component only.
        
        Makes predictions using only the learned fixed effects model, ignoring
        random effects. For predictions that include random effects, use
        compute_random_effects() and add the total effect to predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Covariates for prediction, shape (n, p).

        Returns
        -------
        predictions : np.ndarray
            Predicted values from fixed effects only, shape (n, m).
        
        Notes
        -----
        Current implementation does not support random effects in prediction.
        To obtain predictions including random effects:
        
        1. Call compute_random_effects() to get random effect estimates
        2. Add total_effect to fixed effect predictions
        
        Examples
        --------
        >>> # Predict with fixed effects only
        >>> y_pred = model.predict(X_new)
        
        >>> # Predict with both fixed and random effects
        >>> y_fixed = model.predict(X_new)
        >>> _, total_re, _ = model.compute_random_effects(X_new, y_new, groups_new)
        >>> y_pred_full = y_fixed + total_re.reshape((-1, model.m))
        """
        return self.fe_model.predict(X)
