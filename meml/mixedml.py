from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from statsmodels.regression.mixed_linear_model import MixedLM

class MEML:
    def __init__(self,
                 fixed_effects_model: RegressorMixin,
                 max_iter: Optional[int] = 10,
                 gll_limit: Optional[float] = 0.001):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.gll_limit = gll_limit

    def predict(self, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        Predict using trained mixed effect model.
        if groups are known, the random effect are added to the predictions.
            x : Explanatory covariates
            groups : groupsing variable
            z : Random effect covariates
        """
        self._check_inputs(x, groups)
        y_pred = self.fe_model.predict(x)

        z = np.ones((len(groups), 1))
        common_groups = np.intersect1d(np.unique(self.groups), np.unique(groups), assume_unique=True)
        for g in common_groups:
            group_mask = (groups == g)
            mask_re = (self.groups == g)
            re_i = self.resid_re[mask_re][0]
            y_pred[group_mask] += (z[group_mask] @ re_i).ravel()
        return y_pred

    def fit(self, x: np.ndarray, groups: np.ndarray, y: np.ndarray,
            x_val: Optional[np.ndarray] = None, groups_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, method: Optional[str] = None):
        """
        Fit the mixed effect model using Expectation-Maximization algorithm
        TODO: For now it considers one random effect and one response variable
            x : Explanatory covariates
            groups : grouping variable
            z : Random effect covariates (e.g. unit array (n_obs, 1) for random intercept)
            y : Response variable
            *_val : Respective validation inputs
        """
        self._check_inputs(x, groups, y, x_val, groups_val, y_val)
        self._initialize_em_algorithm(groups, y)
        for iteration in range(1, self.max_iter + 1):
            # Expectation step: updating fixed effects and estimating observations
            y_fe = self.update_y_fe(y)
            y_pred = self.fe_model.fit(x, y_fe).predict(x)
            # Maximization step: updating residual and variance components
            if method == 'mixedlm':
                self.update_residual_variance_mixedlm(y, groups, y_pred)
            else:
                self.update_residual_variance(y, y_pred)
            gll = self.update_gll()
            self.track_variables(gll)
            if x_val is not None:
                mse_val = self._perform_validation(x_val, groups_val, y_val) 
            self._print_iteration_info(mse_val, iteration, gll)
            if self._is_converged(gll):
                break
        return self
    
    def _check_inputs(self, x: np.ndarray, groups: np.ndarray, y: Optional[np.ndarray] = None,
                     x_val: Optional[np.ndarray] = None, groups_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None):
        if x.ndim != 2 or x.dtype != float: raise ValueError("x must be 2D float")
        if groups.ndim != 1: raise ValueError("groups must be 1D")
        if y is not None and (y.ndim != 1 or y.dtype != float): raise ValueError("y must be 1D float")
        if x_val is not None:
            if x_val.ndim != 2 or x_val.dtype != float: raise ValueError("x_val must be 2D float")
            if groups_val.ndim != 1: raise ValueError("groups_val must be 1D")
            if y_val.ndim != 1 or y_val.dtype != float: raise ValueError("y_val must be 1D float")
    
    def _initialize_em_algorithm(self, groups, y):
        """Initialize variables for the EM algorithm."""
        self.groups = groups
        unique_groups = np.unique(groups)
        self.n_groups = len(unique_groups)
        self.n_obs = len(y)
        self.n_re = 1  #TODO Default: one random effect random intercept only
        self.z = np.ones((self.n_obs, self.n_re))
        self.group_indices = [np.where(groups == g)[0] for g in unique_groups]
        self.resid_re = np.zeros((self.n_obs, self.n_re, 1))
        self.resid_unexp = np.zeros_like(y)
        self.var_re = np.ones((self.n_re, self.n_re)) * 0.1
        self.var_unexp = 0.2
        self.var_re_history = []
        self.var_unexp_history = []
        self.gll_history = []
        self.valid_history = []
        return self

    def update_y_fe(self, y):
        """
        Update the fixed effect residuals (observations - random effects)
        """
        y_fe = y.copy()
        for group in self.group_indices:
            z_i = self.z[group]
            re_i = self.resid_re[group[0]]  # RE is unique for each group
            y_fe[group] -= (z_i @ re_i).ravel()
        return y_fe
    
    def update_residual_variance_mixedlm(self, y, groups, y_pred):
        """
        Update the residual and variance components using MixedLM from statsmodels
        """
        residuals = y - y_pred
        # Define mixedlm without explanatory variables
        result = MixedLM(residuals, np.ones_like(residuals), groups).fit()
        for group, effect in result.random_effects.items():
            mask = (groups == group)
            self.resid_re[mask] = effect.iloc[0]
        self.resid_unexp[...] = result.resid
        self.var_re[...] = result.cov_re
        self.var_unexp = result.scale
        return self

    def update_residual_variance(self, y, y_pred):
        """
        Update the residual and variance components using the EM algorithm
        """
        var_re_sum = np.zeros_like(self.var_re)
        var_unexp_sum = 0.0
        for group in self.group_indices:
            y_i = y[group]
            z_i = self.z[group]
            n_i = len(group)
            ident_i = np.eye(n_i)
            y_pred_i = y_pred[group]
            # Use solver technique instead of explicit matrix inversion
            V_i = z_i @ self.var_re @ z_i.T + self.var_unexp * ident_i
            x = np.linalg.solve(V_i, y_i - y_pred_i)
            # Compute random effect and unexplained residuals
            re_i = self.var_re @ z_i.T @ x
            eps_i = y_i - y_pred_i - z_i @ re_i

            self.resid_re[group] = re_i
            self.resid_unexp[group] = eps_i

            var_unexp_sum += (eps_i.T @ eps_i + self.var_unexp *
                              (n_i - self.var_unexp * np.trace(np.linalg.solve(V_i, ident_i))))
            var_re_sum += (re_i @ re_i.T +
                                (self.var_re - self.var_re @ z_i.T @ np.linalg.solve(V_i, z_i @ self.var_re)))

        self.var_re[...] = var_re_sum / self.n_groups
        self.var_unexp = var_unexp_sum / self.n_obs
        return self

    def update_gll(self):
        """
        Updat the (negative) gll based on variance and residual components
        """
        gll_sum = 0.0
        logdet_var_re_total = np.linalg.slogdet(self.var_re)[1] * self.n_groups  # using total to avoid re-calculating in the loop
        logdet_var_unexp_total = np.log(self.var_unexp) * self.n_obs  # scaled identity matrix: easy inversion and determinant
        for group in self.group_indices:
            re_i = self.resid_re[group[0]]
            eps_i = self.resid_unexp[group]
            eps_term = (eps_i.T @ eps_i) / self.var_unexp
            re_term = (re_i.T @ np.linalg.solve(self.var_re, re_i)).item()
            gll_sum += eps_term + re_term
        return gll_sum + logdet_var_re_total + logdet_var_unexp_total + np.log(2 * np.pi) * (self.n_obs + self.n_re)  # -2 * log-likelihood

    def track_variables(self, gll):
        """
        Track changes in MEML variables
        """
        self.var_re_history.append(self.var_re.copy())
        self.var_unexp_history.append(self.var_unexp)
        self.gll_history.append(gll)
        return self
    
    def _perform_validation(self, x_val, groups_val, y_val):
        y_val_pred = self.predict(x_val, groups_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        self.valid_history.append(mse_val)
        return mse_val

    def _print_iteration_info(self, mse_val, iteration, gll):
        print("{:-<20}{:-^20}{:->20}\n".format(f"GLL: {gll:.4f}",
                                               f"MSE: {mse_val:.4f}" if mse_val else "None",
                                               f"Iteration: {iteration}"))

    def _is_converged(self, gll) -> bool:
        if self.gll_limit and len(self.gll_history) > 1:
            err = np.abs((gll - self.gll_history[-2]) / self.gll_history[-2])
            if err < self.gll_limit:
                print("{:-<50}".format(f"GLL converged: {err:.4f}<{self.gll_limit}"))
                return True
        return False

    def summary(self):
        """
        Plots of variance components and residuals and validation per iteration
        """
        with plt.rc_context({
            'font.family': 'Times New Roman',
            'font.size': 9,

            'lines.linewidth': 0.5,
            'lines.markersize': 1,

            'axes.titlesize': 'medium',
            'axes.linewidth': 0.2,

            'xtick.major.width': 0.2,
            'ytick.major.width': 0.2,
            'xtick.minor.width': 0.15,
            'ytick.minor.width': 0.15,

            'legend.framealpha': 1.0,
            'legend.frameon': False,

            'figure.dpi': 900,
            'figure.figsize': (3.937, 3.1496),  # 10 cm by 8 cm
            'figure.constrained_layout.use': True,

            'patch.linewidth': 0.5,}):
            fig, axes = plt.subplots(3, 2, figsize=(12/2.54, 12/2.54), layout='constrained')
            axes = axes.flatten()

            # Generalized Log-Likelihood
            axes[0].plot(self.gll_history, marker='o')
            axes[0].set_ylabel("GLL")
            axes[0].set_xlabel("Iteration")
            axes[0].text(0.95, 0.95, f'GLL = {self.gll_history[-1]:.2f}', transform=axes[0].transAxes, va='top', ha='right')

            # Validation MSE
            if self.valid_history:
                axes[1].plot(self.valid_history, marker='o')
                axes[1].text(0.95, 0.95, f'MSE = {self.valid_history[-1]:.2f}', transform=axes[1].transAxes, va='top', ha='right')
            axes[1].set_ylabel("MSE")
            axes[1].set_xlabel("Iteration")

            # Fixed effect metrics
            axes[2].plot(self.var_unexp_history, label="Unexplained variance", marker='o')
            axes[2].set_ylabel("Unexplained variance")
            axes[2].set_xlabel("Iteration")
            axes[2].text(0.95, 0.95, fr'$\mathregular{{\sigma}}$ = {self.var_unexp_history[-1]:.2f}', transform=axes[2].transAxes, va='top', ha='right')

            # Random effect metrics
            det_re_history = [np.linalg.det(x) for x in self.var_re_history]
            trace_re_history = [np.trace(x) for x in self.var_re_history]
            axes[3].plot(det_re_history, label="|Random effect|", marker='o', zorder=2)
            axes[3].plot(trace_re_history, label="trace(Random effect)", marker='o', zorder=1)
            axes[3].set_ylabel("Random effect variance")
            axes[3].set_xlabel("Iteration")
            axes[3].text(0.95, 0.95, fr'$\mathregular{{\tau}}$ = {self.var_re_history[-1].item():.2f}', transform=axes[3].transAxes, va='top', ha='right')

            # Unexplained residuals distribution
            axes[4].hist(self.resid_unexp)
            axes[4].set_ylabel('Frequency')
            axes[4].set_xlabel("Unexplained residual")

            # Random effect residuals distribution
            axes[5].hist(self.resid_re.ravel())
            axes[5].set_ylabel('Frequency')
            axes[5].set_xlabel("Random effect residual")
            for ax in axes:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True)) if ax != axes[-1] and ax != axes[-2] else None
                ax.minorticks_on()
                ax.grid(True, which='major', linewidth=0.15)
                ax.grid(True, which='minor', linestyle=':', linewidth=0.1)
            plt.show()
