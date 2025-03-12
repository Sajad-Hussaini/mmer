import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from statsmodels.regression.mixed_linear_model import MixedLM

class MEML:
    def __init__(self,
                 fixed_effects_model: object = None,
                 max_iter: int = 10,
                 gll_stop: float = None,
                 tuning: bool = False) -> None:
        """
        Initialize the Mixed Effect Model.

        Parameters
        ----------
        fixed_effects_model : a regressor for fixed effects
        max_iter : Max Number of iteration for EM algorithm
        gll_stop : a stop threshold for GLL.
        tuning : flag to tune hyperparameters of ML model

        var_re: random effect covariance matrix (i.e., between event variance)
        var_unexp: unexplained variance (i.e., within event variance)
        """
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.gll_stop = gll_stop
        self.tuning = tuning
        self.var_unexp_history = []
        self.var_re_history = []
        self.gll_history = []
        self.valid_history = []

    def predict(self, X: np.ndarray, cluster: np.array, Z: np.ndarray) -> np.array:
        """
        Predict using trained mixed effect model.
        if clusters are known, the random effect are added to the predictions.

        Parameters
        ----------
        X : Explanatory covariates
        cluster : Clustering variable
        Z : Random effect covariates

        Returns
        -------
        y_pred : Predicted values
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        cluster = cluster.reshape(-1) if cluster.ndim == 2 else cluster

        y_pred = self.fe_model.predict(X)
        common_clusters = np.intersect1d(np.unique(self.cluster), np.unique(cluster))

        for cluster_i in common_clusters:
            mask_i = (cluster == cluster_i).reshape(-1)
            mask_re = (self.cluster == cluster_i).reshape(-1)
            re_i = (self.resid_re[mask_re][0]).reshape(-1, 1)
            y_pred[mask_i] += (Z[mask_i] @ re_i).reshape(-1)
        return y_pred
    
    def fit_lme(self, X: np.ndarray, cluster: np.array, Z: np.ndarray, y: np.array,
            X_val: np.ndarray = None, cluster_val: np.array = None, Z_val: np.ndarray = None, y_val: np.array = None):
        """
        Fit the mixed effect model using Expectation-Maximization algorithm
        TODO: For now one random effect, one response variable
            So, reshape(-1) is used to make them 1D arrays.

        Parameters
        ----------
        X : Explanatory covariates
        cluster : Clustering variable
        Z : Random effect covariates
        y : Response variable
        *_val : Respective validation inputs

        Returns
        -------
        TYPE MEML model
        """
        Z = Z.astype(float)  # float
        y = y.astype(float)  # float
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        y = y.reshape(-1) if y.ndim == 2 else y
        cluster = cluster.reshape(-1) if cluster.ndim == 2 else cluster
        y_val = y_val.reshape(-1) if y_val is not None and y_val.ndim == 2 else y_val
        self.cluster = cluster
        unique_clusters = np.unique(cluster)
        self.n_clusters = len(unique_clusters)
        self.n_obs = X.shape[0]
        self.n_re = Z.shape[1]
        self.mask_by_cluster = [(cluster == cluster_i).reshape(-1)  for cluster_i in unique_clusters]

        # Intialize EM algorithm
        iteration = 0
        resid_re = np.zeros_like(Z)
        var_unexp = 1.0  # float not integer
        var_re = np.ones((self.n_re, self.n_re))
        self.best_fe_model(X, y) if self.tuning else None
        stop_flag = False
        while iteration < self.max_iter and not stop_flag:
            iteration += 1
            # E- step: Update fixed effects (tuning, prediction)
            y_fe = self.update_data_fe(y, Z, resid_re)
            # self.best_fe_model(X, y_fe) if self.tuning else None
            y_pred = self.fe_model.fit(X, y_fe).predict(X)

            residuals = y - y_pred
            # Define model: intercept-only (np.ones for fixed effect)
            X_dummy = np.ones_like(residuals)  # Fixed effect (just intercept)
            
            # Fit mixed-effects model
            model = MixedLM(residuals, X_dummy, groups=cluster)
            result = model.fit()

            # Between-cluster residuals as array
            between_residuals = np.zeros_like(residuals, dtype=float)
            for group, effect in result.random_effects.items():
                mask = (cluster == group)
                between_residuals[mask] = effect.iloc[0]
            
            resid_re = between_residuals  # Dict of cluster effects
            resid_re = resid_re[:, None]
            resid_unexp = result.resid# + result.fe_params[0]  # Add intercept back  # Array of within-cluster residuals
            var_re = result.cov_re  # Assuming one random effect
            var_unexp = result.scale  # Residual variance

            # M-step: update random effects and variance components
            # resid_re, resid_unexp, var_unexp, var_re = self.update_residual_variance(y, Z, resid_re, var_unexp, var_re, y_pred)

            # update GLL (minimizing negative value, potential use is to minimize gll objective function)
            gll = self.update_gll(y, resid_re, resid_unexp, var_unexp, var_re)

            # Track evolution of the EM algorithm
            self.update_variable_history(resid_re, resid_unexp, var_unexp, var_re, gll)

            # Validation using MSE
            if X_val is not None and isinstance(X_val, np.ndarray):
                y_val_pred = self.predict(X_val, cluster_val, Z_val)
                mse_val = mean_squared_error(y_val, y_val_pred)
                self.valid_history.append(mse_val)
            else:
                mse_val = None
            print("{:-<20}{:-^20}{:->20}\n".format(f"GLL: {gll:.4f}",
                                                   f"MSE validation: {mse_val:.4f}" if mse_val is not None else "No validation",
                                                   f"at iteration: {iteration}"))
            # Stop criterion
            if self.gll_stop and len(self.gll_history) > 1 and \
                (err := np.abs((gll - self.gll_history[-2]) / self.gll_history[-2])) < self.gll_stop:
                print("{:-<50}".format(f"GLL converged: {err:.4f}<{self.gll_stop}"))
                stop_flag = True
        return self

    def fit(self, X: np.ndarray, cluster: np.array, Z: np.ndarray, y: np.array,
            X_val: np.ndarray = None, cluster_val: np.array = None, Z_val: np.ndarray = None, y_val: np.array = None):
        """
        Fit the mixed effect model using Expectation-Maximization algorithm
        TODO: For now one random effect, one response variable
            So, reshape(-1) is used to make them 1D arrays.

        Parameters
        ----------
        X : Explanatory covariates
        cluster : Clustering variable
        Z : Random effect covariates
        y : Response variable
        *_val : Respective validation inputs

        Returns
        -------
        TYPE MEML model
        """
        Z = Z.astype(float)  # float
        y = y.astype(float)  # float
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        y = y.reshape(-1) if y.ndim == 2 else y
        cluster = cluster.reshape(-1) if cluster.ndim == 2 else cluster
        y_val = y_val.reshape(-1) if y_val is not None and y_val.ndim == 2 else y_val
        self.cluster = cluster
        unique_clusters = np.unique(cluster)
        self.n_clusters = len(unique_clusters)
        self.n_obs = X.shape[0]
        self.n_re = Z.shape[1]
        self.mask_by_cluster = [(cluster == cluster_i).reshape(-1)  for cluster_i in unique_clusters]

        # Intialize EM algorithm
        iteration = 0
        resid_re = np.zeros_like(Z)
        var_unexp = 1.0  # float not integer
        var_re = np.ones((self.n_re, self.n_re))
        self.best_fe_model(X, y) if self.tuning else None
        stop_flag = False
        while iteration < self.max_iter and not stop_flag:
            iteration += 1
            # E- step: Update fixed effects (tuning, prediction)
            y_fe = self.update_data_fe(y, Z, resid_re)
            # self.best_fe_model(X, y_fe) if self.tuning else None
            y_pred = self.fe_model.fit(X, y_fe).predict(X)

            # M-step: update random effects and variance components
            resid_re, resid_unexp, var_unexp, var_re = self.update_residual_variance(y, Z, resid_re, var_unexp, var_re, y_pred)

            # update GLL (minimizing negative value, potential use is to minimize gll objective function)
            gll = self.update_gll(y, resid_re, resid_unexp, var_unexp, var_re)

            # Track evolution of the EM algorithm
            self.update_variable_history(resid_re, resid_unexp, var_unexp, var_re, gll)

            # Validation using MSE
            if X_val is not None and isinstance(X_val, np.ndarray):
                y_val_pred = self.predict(X_val, cluster_val, Z_val)
                mse_val = mean_squared_error(y_val, y_val_pred)
                self.valid_history.append(mse_val)
            else:
                mse_val = None
            print("{:-<20}{:-^20}{:->20}\n".format(f"GLL: {gll:.4f}",
                                                   f"MSE validation: {mse_val:.4f}" if mse_val is not None else "No validation",
                                                   f"at iteration: {iteration}"))
            # Stop criterion
            if self.gll_stop and len(self.gll_history) > 1 and \
                (err := np.abs((gll - self.gll_history[-2]) / self.gll_history[-2])) < self.gll_stop:
                print("{:-<50}".format(f"GLL converged: {err:.4f}<{self.gll_stop}"))
                stop_flag = True
        return self

    def update_data_fe(self, y, Z, resid_re):
        """
        Update the data for fixed effect regression that is (y-Zb)
        """
        y_fe_update = np.zeros_like(y)
        for mask_cluster_i in self.mask_by_cluster:
            y_i = y[mask_cluster_i]  # (ni,)
            Z_i = Z[mask_cluster_i]  # (ni, n_re)
            re_i = (resid_re[mask_cluster_i][0]).reshape(-1, 1) # (n_re, 1)
            y_fe_update[mask_cluster_i] = y_i - (Z_i @ re_i).reshape(-1)
        return y_fe_update

    def update_residual_variance(self, y, Z, resid_re, var_unexp, var_re, y_pred):
        """
        Update the residual and variance components based on previous values
        """
        resid_re_update = np.zeros_like(Z)
        resid_unexp_update = np.zeros_like(y)
        var_unexp_sum = np.zeros_like(var_unexp)
        var_re_sum = np.zeros_like(var_re)
        for mask_cluster_i in self.mask_by_cluster:
            y_i = y[mask_cluster_i]  # (ni,)
            Z_i = Z[mask_cluster_i]  # (ni, n_re)
            n_i = sum(mask_cluster_i)
            I_i = np.eye(n_i)

            y_pred_i = y_pred[mask_cluster_i]  # (ni,)
            V_inv_i = np.linalg.inv(Z_i @ var_re @ Z_i.T + var_unexp * I_i)  # (ni, ni)

            # Compute random effect and unexplained residuals
            re_i = (var_re @ Z_i.T @ V_inv_i @ (y_i - y_pred_i)).reshape(-1, 1)  # (n_re, 1)
            resid_re_update[mask_cluster_i] = re_i
            eps_i = y_i - y_pred_i - (Z_i @ re_i).reshape(-1)  # (ni,)
            resid_unexp_update[mask_cluster_i] = eps_i
            # no effect .T but it's ok
            var_unexp_sum += (eps_i.T @ eps_i +
                               var_unexp * (n_i - var_unexp * np.trace(V_inv_i)))
            var_re_sum += (re_i @ re_i.T +
                                (var_re - var_re @ Z_i.T @ V_inv_i @ Z_i @ var_re))

        var_unexp_update = var_unexp_sum / self.n_obs
        var_re_update = var_re_sum / self.n_clusters
        return resid_re_update, resid_unexp_update, var_unexp_update, var_re_update

    def update_gll(self, y, resid_re, resid_unexp, var_unexp, var_re):
        """
        Update the (negative) gll based on variance and residual components
        """
        gll_sum = 0
        for mask_cluster_i in self.mask_by_cluster:
            n_i = sum(mask_cluster_i)
            var_unexp_mat_i = var_unexp * np.eye(n_i)
            re_i = resid_re[mask_cluster_i][0]
            eps_i = resid_unexp[mask_cluster_i]

            logdet_var_re = np.linalg.slogdet(var_re)[1]
            logdet_var_unexp_mat_i = np.linalg.slogdet(var_unexp_mat_i)[1]
            # minimize below that is (the actual GLL x -1 x 2 and droping constant terms)
            gll_sum += (eps_i.T @ np.linalg.inv(var_unexp_mat_i) @ eps_i
                    + re_i.T @ np.linalg.inv(var_re) @ re_i
                    + logdet_var_re
                    + logdet_var_unexp_mat_i)
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

    def best_fe_model(self, X, y):
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
                'max_samples': uniform(0.5, 0.4)              # Fraction of samples to draw from X to train each base estimator
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
        elif fe_model_name == 'XGBRegressor':
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
                           scoring='neg_mean_squared_error', n_jobs=-1, n_iter=20).fit(X, y)
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
            axes[5].hist(self.resid_re)
            axes[5].set_ylabel('Frequency')
            axes[5].set_xlabel("Random effect residual")
            for ax in axes:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True)) if ax != axes[-1] and ax != axes[-2] else None
                ax.grid(True, alpha=0.25)
                ax.minorticks_on()
            plt.show()
