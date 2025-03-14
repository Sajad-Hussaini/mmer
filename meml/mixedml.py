from typing import Optional, Tuple, Union
from sklearn.base import RegressorMixin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from statsmodels.regression.mixed_linear_model import MixedLM

class MEML:
    def __init__(self,
                 fixed_effects_model: RegressorMixin = None,
                 max_iter: int = 10,
                 gll_limit: Optional[float] = None,
                 tuning: bool = False):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.gll_limit = gll_limit
        self.tuning = tuning
        self.var_unexp_history: list = []
        self.var_re_history: list = []
        self.gll_history: list = []
        self.valid_history: list = []

    def predict(self, x: np.ndarray, cluster: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Predict using trained mixed effect model.
        if clusters are known, the random effect are added to the predictions.

        Parameters
        ----------
        x : Explanatory covariates
        cluster : Clustering variable
        z : Random effect covariates

        Returns
        -------
        y_pred : Predicted values
        """
        self._check_input(x, cluster, z)
        y_pred = self.fe_model.predict(x)
        common_clusters = np.intersect1d(np.unique(self.cluster), np.unique(cluster), assume_unique=True)

        for cluster_i in common_clusters:
            mask_i = (cluster == cluster_i)
            mask_re = (self.cluster == cluster_i)
            re_i = self.resid_re[mask_re][0]
            y_pred[mask_i] += (z[mask_i] @ re_i).ravel()
        return y_pred

    def _check_input(self, x: np.ndarray, cluster: np.ndarray, z: np.ndarray, y: Optional[np.ndarray] = None,
                     x_val: Optional[np.ndarray] = None, cluster_val: Optional[np.ndarray] = None,
                     z_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        assert x.ndim == 2, "x must be 2D"
        assert x.dtype == float, "x must be float"
        assert z.ndim == 2, "z must be 2D"
        assert z.dtype == float, "z must be float"
        assert cluster.ndim == 1, "cluster must be 1D"
        if y is not None:
            assert y.ndim == 1, "y must be 1D"
            assert y.dtype == float, "y must be float"
        if x_val is not None:
            assert x_val.ndim == 2, "x_val must be 2D"
            assert x_val.dtype == float, "x_val must be float"
            assert cluster_val.ndim == 1, "cluster_val must be 1D"
            assert z_val.ndim == 2, "z_val must be 2D"
            assert z_val.dtype == float, "z_val must be float"
            assert y_val.ndim == 1, "y_val must be 1D"
            assert y_val.dtype == float, "y_val must be float"

    def fit(self, x: np.ndarray, cluster: np.ndarray, z: np.ndarray, y: np.ndarray,
            x_val: Optional[np.ndarray] = None, cluster_val: Optional[np.ndarray] = None,
            z_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, method = None):
        """
        Fit the mixed effect model using Expectation-Maximization algorithm
        TODO: For now it considers one random effect and one response variable

            x : Explanatory covariates
            cluster : Clustering variable
            z : Random effect covariates
            y : Response variable
            *_val : Respective validation inputs
        """
        self._check_input(x, cluster, z, y, x_val, cluster_val, z_val, y_val)
        self.cluster = cluster
        unique_clusters = np.unique(cluster)
        self.n_clusters = len(unique_clusters)
        self.n_obs = x.shape[0]
        self.n_re = z.shape[1]
        self.mask_by_cluster = [np.isin(cluster, cluster_i) for cluster_i in unique_clusters]

        # Intialize EM algorithm
        iteration = 0
        resid_re = np.zeros((self.n_obs, self.n_re, 1))
        resid_unexp = np.zeros_like(y)
        var_re = np.ones((self.n_re, self.n_re))
        var_unexp = 1.0  # float not integer
        self.best_fe_model(x, y) if self.tuning else None
        stop_flag = False
        while iteration < self.max_iter and not stop_flag:
            iteration += 1
            # E- step: Update fixed effects (tuning, prediction)
            y_fe = self.update_data_fe(y, z, resid_re)
            # self.best_fe_model(x, y_fe) if self.tuning else None
            y_pred = self.fe_model.fit(x, y_fe).predict(x)

            if method == 'mixedlm':
                residuals = y - y_pred
                # Define mixedlm using intercept-only (np.ones for fixed effect)
                x_dummy = np.ones_like(residuals)
                model = MixedLM(residuals, x_dummy, groups=cluster)
                result = model.fit()
                resid_re = np.zeros((self.n_obs, self.n_re, 1))
                for group, effect in result.random_effects.items():
                    mask = (cluster == group)
                    resid_re[mask] = effect.iloc[0]
                resid_unexp = result.resid
                var_re = result.cov_re
                var_unexp = result.scale
            else:
                # M-step: update random effects and variance components
                resid_re, resid_unexp, var_re, var_unexp = self.update_residual_variance(y, z, resid_re, resid_unexp, var_re, var_unexp, y_pred)

            # update GLL (minimizing negative value, potential use is to minimize gll objective function)
            gll = self.update_gll(y, resid_re, resid_unexp, var_unexp, var_re)

            # Track evolution of the EM algorithm
            self.update_variable_history(resid_re, resid_unexp, var_unexp, var_re, gll)

            # Validation using MSE
            if x_val is not None and isinstance(x_val, np.ndarray):
                y_val_pred = self.predict(x_val, cluster_val, z_val)
                mse_val = mean_squared_error(y_val, y_val_pred)
                self.valid_history.append(mse_val)
            else:
                mse_val = None
            print("{:-<20}{:-^20}{:->20}\n".format(f"GLL: {gll:.4f}",
                                                   f"MSE validation: {mse_val:.4f}" if mse_val is not None else "No validation",
                                                   f"at iteration: {iteration}"))
            # Stop criterion
            if self.gll_limit and len(self.gll_history) > 1 and \
                (err := np.abs((gll - self.gll_history[-2]) / self.gll_history[-2])) < self.gll_limit:
                print("{:-<50}".format(f"GLL converged: {err:.4f}<{self.gll_limit}"))
                stop_flag = True
        return self

    def update_data_fe(self, y, z, resid_re):
        """
        Update the data for fixed effect regression that is (y-zb)
        """
        y_fe_update = y.copy()
        for mask_i in self.mask_by_cluster:
            z_i = z[mask_i]
            re_i = resid_re[mask_i][0]  # Because re is the same for a cluster
            y_fe_update[mask_i] -= (z_i @ re_i).ravel()  # In-place subtraction
        return y_fe_update

    def update_residual_variance(self, y, z, resid_re, resid_unexp, var_re, var_unexp, y_pred):
        """
        Update the residual and variance components based on previous values
        """
        var_unexp_sum = np.zeros_like(var_unexp)
        var_re_sum = np.zeros_like(var_re)
        for mask_i in self.mask_by_cluster:
            y_i = y[mask_i]
            z_i = z[mask_i]
            n_i = mask_i.sum()
            ident_i = np.eye(n_i)
            y_pred_i = y_pred[mask_i]

            # Use solver technique instead of explicit matrix inversion
            V_i = z_i @ var_re @ z_i.T + var_unexp * ident_i
            x = np.linalg.solve(V_i, y_i - y_pred_i)

            # Compute random effect and unexplained residuals
            re_i = var_re @ z_i.T @ x
            eps_i = y_i - y_pred_i - z_i @ re_i

            resid_re[mask_i] = re_i[:, None]  # check if we need [:, None]
            resid_unexp[mask_i] = eps_i

            var_unexp_sum += (eps_i.T @ eps_i + var_unexp *
                              (n_i - var_unexp * np.trace(np.linalg.solve(V_i, ident_i))))
            var_re_sum += (re_i @ re_i.T +
                                (var_re - var_re @ z_i.T @ np.linalg.solve(V_i, z_i @ var_re)))

        var_unexp = var_unexp_sum / self.n_obs
        var_re = var_re_sum / self.n_clusters
        return resid_re, resid_unexp, var_re, var_unexp

    def update_gll(self, y, resid_re, resid_unexp, var_unexp, var_re):
        """
        Update the (negative) gll based on variance and residual components
        """
        gll_sum = 0.0
        logdet_var_re = np.linalg.slogdet(var_re)[1]  # just add to each iteration instead of re-calculating
        for mask_i in self.mask_by_cluster:
            n_i = mask_i.sum()
            re_i = resid_re[mask_i][0]
            eps_i = resid_unexp[mask_i]
            # Becasue unexplained variance is always a scaled identity matrix A=xI simpy A^-1 = I/x
            # also log determinant is just n*log(x) for A=xI
            logdet_var_unexp = n_i * np.log(var_unexp)
            eps_term = (eps_i.T @ eps_i) / var_unexp  # always scalar operation
            re_term = (re_i.T @ np.linalg.solve(var_re, re_i)).item()  # Direct inversion for small var_re
            gll_sum += eps_term + re_term + logdet_var_re + logdet_var_unexp
        return gll_sum

    def update_variable_history(self, resid_re, resid_unexp, var_unexp, var_re, gll):
        """
        Update the history of MEML variables to track changes
        """
        self.resid_re = resid_re  # residual explained by random effect
        self.resid_unexp = resid_unexp  # unexplained residual
        self.var_unexp = var_unexp
        self.var_re = np.diag(var_re)[0]  # single re
        self.var_unexp_history.append(var_unexp)
        self.var_re_history.append(var_re)
        self.gll_history.append(gll)
        return self

    def best_fe_model(self, x, y):
        """
        Tune hyperparameters of the fixed effect regressor
        """
        fe_model_name = type(self.fe_model).__name__
        if fe_model_name == 'RandomForestRegressor':
            from scipy.stats import randint, uniform
            param_dist = {
                'n_estimators': randint(5, 400),              # Number of trees in the forest
                'max_depth': randint(2, 15),                  # Maximum depth of the trees
                'min_samples_split': randint(2, 18),          # Minimum number of samples required to split an internal node
                'min_samples_leaf': randint(1, 10),           # Minimum number of samples required to be at a leaf node
                'max_samples': uniform(0.5, 0.4)              # Fraction of samples to draw from x to train each base estimator
            }
        elif fe_model_name == 'MLPRegressor':
            # param_dist = {
            #     'hidden_layer_sizes': [(5,), (8,), (12,), (18,), (25,), (50,)]}
            param_dist = {
                'hidden_layer_sizes': [(5,), (10,), (5, 5), (5, 5, 5), (100, 100), (50, 50, 50)],
                'activation': ['relu', 'tanh', 'logistic'],  # 'logistic' will be tested again here
                'solver': ['adam', 'sgd', 'lbfgs'],
                'learning_rate': ['constant', 'adaptive']}
        elif fe_model_name == 'CatBoostRegressor':
            param_dist = {
                'iterations': (5, 300),
                'learning_rate': (0.01, 0.5),
                'depth': (2, 15),
                'l2_leaf_reg': (1, 7),
                'bagging_temperature': (0.5, 1.5)}
        elif fe_model_name == 'GradientBoostingRegressor':
            param_dist = {
                'n_estimators': (5, 400),
                'learning_rate': (0.01, 0.5),
                'max_depth': (2, 15),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 8)}
        elif fe_model_name == 'xGBRegressor':
            param_dist = {
                'n_estimators': (5, 400),
                'max_depth': (2, 15),
                'learning_rate': (0.001, 0.1),
                'min_child_weight': (1, 5),
                'subsample': (0.5, 0.9),
                'colsample_bytree': (0.5, 0.9)}
        elif fe_model_name == 'LGBMRegressor':
            param_dist = {
                'n_estimators': (5, 400),
                'learning_rate': (0.001, 0.1),
                'max_depth': (2, 15),
                'min_child_samples': (5, 40),
                'subsample': (0.7, 0.9),
                'colsample_bytree': (0.6, 0.9)}
        else:
            raise ValueError("Unknown regressor for hyperparameter tuning.")
        opt = RandomizedSearchCV(self.fe_model, param_dist, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1, n_iter=20).fit(x, y)
        self.fe_model, self.fe_model_params = opt.best_estimator_, opt.best_params_
        return self

    def summary(self):
        """
        Plots of variance components and residuals and validation per iteration
        """
        with plt.rc_context({
                'figure.dpi': 600,
                'lines.linewidth': 0.5,
                'lines.markersize': 1,
                'font.size': 9,
                'axes.labelsize': 9,
                'legend.fontsize': 9,
                'font.family': 'Times New Roman',
                'axes.linewidth': 0.25,
                'xtick.major.width': 0.25,
                'xtick.minor.width': 0.25,
                'ytick.major.width': 0.25,
                'ytick.minor.width': 0.25}):
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
                ax.grid(True, alpha=0.25)
                ax.minorticks_on()
            plt.show()
