import pickle
import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import solve
from sklearn.base import RegressorMixin
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from .operator import VLinearOperator, ResidualPreconditioner, compute_cov_correction
from ..lanczos_algorithm import slq
from .random_effect import RealizedRandomEffect, RealizedResidual
from .terms import RandomEffectTerm, ResidualTerm

class MixedEffectRegressor:
    """
    Multivariate Mixed Effects Regression.
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

        self.log_likelihood = []
        self._is_converged = False
        
        # State: Terms
        self.random_effect_terms = None # List[RandomEffectTerm]
        self.residual_term = None # ResidualTerm

        # Training history / Best state
        self._best_log_likelihood = -np.inf
        self._best_re_covs = None
        self._best_resid_cov = None
        self._best_fe_model = None
        
    def _prepare_terms(self, m: int, groups: np.ndarray, random_slopes: tuple[list[int] | None] | None):
        """
        Initialize the RandomEffectTerm and ResidualTerm objects if they don't exist.
        """
        k = groups.shape[1]
        
        # 1. Initialize Random Structure Config
        if random_slopes is None:
            config_random_slopes = tuple([None] * k)
        elif len(random_slopes) != k:
             raise ValueError(f"Length of random_slopes ({len(random_slopes)}) must match number of groups ({k}).")
        else:
            config_random_slopes = random_slopes
            
        # 2. Create Terms
        self.random_effect_terms = []
        for i, slope_cols in enumerate(config_random_slopes):
            term = RandomEffectTerm(group_id=i, covariates_id=slope_cols, m=m)
            self.random_effect_terms.append(term)
            
        self.residual_term = ResidualTerm(m=m)
        self.k = k # number of groups
        self.m = m # number of outputs
        self.random_slopes = config_random_slopes

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                     validation_split: float = 0.0, validation_group: int = 0):
        """
        Prepare realization of random effects and residual for the given data.
        """
        n, m = y.shape
        self.n = n # Current batch N
        
        # Setup Validation Split
        if validation_split > 0:
            main_group = groups[:, validation_group]
            gss = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
            self.train_idx, self.val_idx = next(gss.split(X, y, groups=main_group))
            self.has_validation = True
        else:
            self.train_idx = np.arange(n)
            self.val_idx = None
            self.has_validation = False
            
        # Instantiate Realized Random Effects (Transient)
        realized_effects = tuple(RealizedRandomEffect(term, X, groups) for term in self.random_effect_terms)
        
        # Instantiate Realized Residual (Transient)
        realized_residual = RealizedResidual(self.residual_term, n)
        
        # Initial Marginal Residual
        marginal_residual = self._compute_marginal_residual(X, y, 0.0)
        
        return marginal_residual, realized_effects, realized_residual

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slopes: None | tuple[list[int]] = None,
            validation_split: float = 0.0, validation_group: int = 0):
        """
        Fit the model.
        """
        n, m = y.shape
        # Initialize terms if new training
        if self.random_effect_terms is None:
            self._prepare_terms(m, groups, random_slopes)
            
        marginal_residual, realized_effects, realized_residual = self.prepare_data(X, y, groups, validation_split, validation_group)
        
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        
        for _ in pbar:
            marginal_residual = self._run_em_iteration(X, y, marginal_residual, realized_effects, realized_residual)
            if self._is_converged:
                pbar.set_description(f"Model Converged | Early stopping.")
                break
                
        from .mixed_result import MixedEffectResults
        return MixedEffectResults(self)

    def _run_em_iteration(self, X, y, marginal_residual, realized_effects, realized_residual):
        total_random_effect, mu, V_op, M_op = self._e_step(marginal_residual, realized_effects, realized_residual)
        
        if self._is_converged:
            return marginal_residual
            
        marginal_residual = self._compute_marginal_residual(X, y, total_random_effect.reshape((self.m, self.n)).T)
        self._m_step(marginal_residual, total_random_effect, mu, realized_effects, realized_residual, V_op, M_op)
        
        return marginal_residual

    def _e_step(self, marginal_residual, realized_effects, realized_residual):
        prec_resid, V_op, M_op = self._solver(marginal_residual, realized_effects, realized_residual)
        
        current_log_lh = self._compute_log_likelihood(marginal_residual, prec_resid, V_op)
        self.log_likelihood.append(current_log_lh)
        
        if len(self.log_likelihood) >= 2:
            self._check_convergence()
            
        if self._is_converged:
             return None, None, None, None
             
        total_random_effect, mu = self._aggregate_random_effects(prec_resid, realized_effects)
        return total_random_effect, mu, V_op, M_op

    def _m_step(self, marginal_residual, total_random_effect, mu, realized_effects, realized_residual, V_op, M_op):
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

    def _solver(self, marginal_residual, realized_effects, realized_residual):
        V_op = VLinearOperator(realized_effects, realized_residual)
        M_op = None
        if self.preconditioner:
            try:
                resid_cov_inv = solve(a=self.residual_term.cov, b=np.eye(self.m), assume_a='pos')
                M_op = ResidualPreconditioner(resid_cov_inv, self.n, self.m)
            except Exception:
                pass
        
        prec_resid, info = cg(A=V_op, b=marginal_residual, M=M_op)
        if info != 0:
            print(f"Warning: CG info={info}")
        return prec_resid, V_op, M_op

    def _compute_marginal_residual(self, X, y, total_random_effect):
        y_adj = y - total_random_effect
        y_adj = y_adj.ravel() if self.m == 1 else y_adj

        if self.has_validation:
            X_train = X[self.train_idx]
            y_adj_train = y_adj[self.train_idx]
            X_val = X[self.val_idx]
            y_adj_val = y_adj[self.val_idx]
            self.fe_model.fit(X_train, y_adj_train, X_val=X_val, y_val=y_adj_val)
        else:
            self.fe_model.fit(X, y_adj)

        fx = self.fe_model.predict(X)
        fx = fx[:, None] if self.m == 1 else fx
        return (y - fx).T.ravel()

    def _compute_log_likelihood(self, marginal_residual, prec_resid, V_op):
        log_det_V = slq.logdet(V_op, self.slq_steps, self.slq_probes, self.n_jobs, self.backend)
        log_likelihood = -(self.m * self.n * np.log(2 * np.pi) + log_det_V + marginal_residual.T @ prec_resid) / 2
        return log_likelihood

    def _aggregate_random_effects(self, prec_resid, realized_effects):
        total_random_effect = np.zeros(self.m * self.n)
        mu = []
        for re in realized_effects:
            mu.append(re._compute_mu(prec_resid))
            total_random_effect += re._map_mu(mu[-1])
        return total_random_effect, tuple(mu)

    def _check_convergence(self):
        change = np.abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2])
        self._is_converged = change <= self.tol
        
        current_log_lh = self.log_likelihood[-1]
        
        # Save best state
        if current_log_lh > self._best_log_likelihood:
            self._best_log_likelihood = current_log_lh
            self._no_improvement_count = 0
            self._best_re_covs = [term.cov.copy() for term in self.random_effect_terms]
            self._best_resid_cov = self.residual_term.cov.copy()
            self._best_fe_model = pickle.loads(pickle.dumps(self.fe_model))
        else:
            self._no_improvement_count += 1
            
        if self._no_improvement_count >= self.patience:
            # Restore best state
            for k, cov in enumerate(self._best_re_covs):
                self.random_effect_terms[k].set_cov(cov)
            self.residual_term.set_cov(self._best_resid_cov)
            self.fe_model = self._best_fe_model
            self._is_converged = True

    # ================= Public Inference Methods =================
    
    def compute_random_effects(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Compute posterior mean of random effects for new data.
        """
        if self.random_effect_terms is None:
            raise RuntimeError("Model is not fitted.")
            
        n, m = y.shape
        realized_effects = tuple(RealizedRandomEffect(term, X, groups) for term in self.random_effect_terms)
        realized_residual = RealizedResidual(self.residual_term, n)
        
        # Predict Fixed Effects
        fx = self.fe_model.predict(X)
        if m == 1 and fx.ndim == 1: fx = fx[:, None]
        
        marginal_resid = (y - fx).T.ravel()
        
        # Solve
        # Use transient solver
        V_op = VLinearOperator(realized_effects, realized_residual)
        
        # Preconditioner
        try:
             resid_cov_inv = solve(a=self.residual_term.cov, b=np.eye(self.m), assume_a='pos')
             M_op = ResidualPreconditioner(resid_cov_inv, n, m)
        except:
             M_op = None
             
        prec_resid, info = cg(A=V_op, b=marginal_resid, M=M_op)
        
        total_re = np.zeros(m * n)
        mu = []
        for re in realized_effects:
             val = re._compute_mu(prec_resid)
             mu.append(val)
             total_re += re._map_mu(val)
             
        resid = marginal_resid - total_re
        return mu, resid

    def predict(self, X: np.ndarray, groups: np.ndarray = None, random_effects: bool = False):
        """
        Predict using fixed effects. 
        """
        return self.fe_model.predict(X)
