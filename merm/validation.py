import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from .style import style

def evaluate_metrics(y_obs, y_pred):
    metrics = {
        'MAE': mean_absolute_error(y_obs, y_pred),
        'MAPE': mean_absolute_percentage_error(y_obs, y_pred),
        'MSE': mean_squared_error(y_obs, y_pred),
        'r2': r2_score(y_obs, y_pred),
        'r': stats.pearsonr(y_obs, y_pred)[0]}
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    return metrics

def print_metrics(metrics_tr, metrics_te):
    print(f"{'Metric':^10}{'Train':^10}{'Test':^10}")
    for metric, tr_value in metrics_tr.items():
        te_value = metrics_te.get(metric, None)
        print(f"{metric:^10}{tr_value:^10.3f}{te_value:^10.3f}")

def model_performance(y_tr_obs, y_tr_pred, y_te_obs, y_te_pred, label):
    metric_tr = evaluate_metrics(y_tr_obs, y_tr_pred)
    metric_te = evaluate_metrics(y_te_obs, y_te_pred)
    print_metrics(metric_tr, metric_te)

    with style():
        plt.figure(figsize=(8/2.54, 8/2.54))
        plt.scatter(y_tr_obs, y_tr_pred, fc='none', ec='tab:blue',
                    label=fr"Train ($R^2$ = {metric_tr['r2']:.3f}, r = {metric_tr['r']:.3f})")
        plt.scatter(y_te_obs, y_te_pred, fc='none', ec='tab:orange',
                    label=f"Test ($R^2$ = {metric_te['r2']:.3f}, r = {metric_te['r']:.3f})")

        lims = plt.gca().get_xlim()
        plt.plot(lims, lims, 'r--', label='Perfect Fit')
        plt.xlabel(f"Observed {label}")
        plt.ylabel(f"Predicted {label}")
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.95), ncol=1)
        plt.show()

def create_data(columns, transformer_x, vary_param, vary_range, fixed_params):
    """
    Create evaluation dataframe with one varying parameter and others fixed.
    
    Parameters:
    - vary_param: str, parameter to vary
    - vary_range: array-like, range of values for the varying parameter
    - fixed_params: dict, fixed parameter values
    """
    df_eval = pd.DataFrame(columns=columns)
    df_eval[vary_param] = vary_range
    
    for param, value in fixed_params.items():
        df_eval[param] = value
    
    return df_eval, transformer_x.transform(df_eval.to_numpy())

def best_fe_model(fe_model, x, y):
        """
        Tune hyperparameters of the fixed effect regressor
        """
        fe_model_name = type(fe_model).__name__
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
            param_dist = {
                'hidden_layer_sizes': [(5,), (8,), (12,), (18,), (25,), (50,), (5, 5), (5, 5, 5), (100, 100), (50, 50, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
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
        opt = RandomizedSearchCV(fe_model, param_dist, cv=5, n_iter=100,
                           scoring='neg_mean_squared_error', n_jobs=-1).fit(x, y)
        return opt.best_estimator_, opt.best_params_

def generate_merm_data(n=1000, M=3, K=2, o_k=[50, 40], p=5, slope_columns=[[0], [0, 2]]):
    """
    Generate synthetic data for a multivariate mixed effects model.

    Parameters:
    - n (int): Number of observations.
    - M (int): Number of responses.
    - K (int): Number of grouping factors.
    - q_k (list): Number of random effect types per group (1 intercept + q_k-1 slopes).
    - o_k (list): Number of levels per group.
    - p (int): Number of predictors.
    - slope_columns (list): List of lists specifying X column indices for random slopes per group.
    - seed (int): Random seed for reproducibility.

    Returns:
    - X (ndarray): (n, p) predictor matrix.
    - Y (ndarray): (n, M) response matrix.
    - groups (ndarray): (n, K) group assignments.
    - rscovariate (list): List of (n, q_k[k]-1) arrays for random slopes.
    - true_phi (ndarray): (M, M) residual covariance matrix.
    - true_tau (list): List of (M*q_k[k], M*q_k[k]) random effect covariance matrices.
    """
    X = np.random.randn(n, p)
    groups = np.zeros((n, K), dtype=int)
    for k in range(K):
        groups[:, k] = np.repeat(np.arange(o_k[k]), n // o_k[k])
        
    # Generate residual covariance (phi: M x M)
    true_phi = np.eye(M) * 0.5
    true_phi[np.triu_indices(M, 1)] = 0.2
    true_phi = true_phi + true_phi.T - np.eye(M) * 0.5

    # Generate random effects covariances (tau_k: M*q_k x M*q_k)
    true_tau = []
    q_k = [1 if slope_columns[i] is None else 1 + len(slope_columns[i]) for i in range(K)]
    # Example: set different diagonal/off-diagonal values for each k
    diag_vals = [0.4 + 0.15 * k for k in range(K)]      # e.g., 0.4, 0.6, 0.8, ...
    off_diag_vals = [0.1 + 0.05 * k for k in range(K)] # e.g., 0.1, 0.15, 0.2, ...
    for k in range(K):
        tau_k = np.eye(M * q_k[k]) * diag_vals[k]
        tau_k[np.triu_indices(M * q_k[k], 1)] = off_diag_vals[k]
        tau_k = tau_k + tau_k.T - np.eye(M * q_k[k]) * diag_vals[k]
        true_tau.append(tau_k)

    # Generate fixed effects
    true_cof = np.random.randn(p, M) * 0.5
    fX = X @ true_cof

    eps = np.random.multivariate_normal(np.zeros(M), true_phi, size=n)
    # eps = np.random.multivariate_normal(np.zeros(M * n), sparse.kron(true_phi, np.eye(n)).toarray()).reshape((n, M), order='F')

    rand_eff = np.zeros((n, M))
    for k in range(K):
        Z_k, q_k_val, o_k_val = random_effect_design_matrix(groups[:, k], X[:, slope_columns[k]] if (slope_columns is not None and slope_columns[k] is not None) else None)
        IM_Z_k = sparse.kron(sparse.eye(M), Z_k)
        b_k = np.random.multivariate_normal(np.zeros(M * q_k[k] * o_k[k]), sparse.kron(true_tau[k], np.eye(o_k[k])).toarray())
        rand_eff += (IM_Z_k @ b_k).reshape((n, M), order='F')

    Y = fX + rand_eff + eps
    return X, Y, groups, slope_columns, true_phi, true_tau, true_cof