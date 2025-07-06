import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from ..visual.style import style

def cov_to_corr(cov):
    """
    Convert covariance matrix to correlation matrix.
    """
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


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

def best_fe_model(fe_model, x, y, cv=5):
    """
    Tune hyperparameters of the fixed effect regressor with exhaustive search.

    Parameters:
    - fe_model: sklearn-compatible regressor instance.
    - x: ndarray, predictor matrix.
    - y: ndarray, response matrix (can be multivariate).
    - cv: int, number of cross-validation folds.

    Returns:
    - best_model: trained model with optimal hyperparameters.
    - best_params: dict, best hyperparameters.
    """
    fe_model_name = type(fe_model).__name__

    # Define hyperparameter grids for supported regressors
    if fe_model_name == 'RandomForestRegressor':
        param_grid = {
            'n_estimators': [50, 100, 200, 400, 800],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    elif fe_model_name == 'MLPRegressor':
        param_grid = {
            'hidden_layer_sizes': [(5,), (10,), (20,), (50,), (100,), (5, 5), (10, 10), (50, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'learning_rate': ['constant', 'adaptive'],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }
    elif fe_model_name == 'CatBoostRegressor':
        param_grid = {
            'iterations': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'bagging_temperature': [0.5, 1.0, 1.5]
        }
    elif fe_model_name == 'GradientBoostingRegressor':
        param_grid = {
            'n_estimators': [50, 100, 200, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'max_depth': [4, 6, 8, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif fe_model_name == 'xGBRegressor':
        param_grid = {
            'n_estimators': [50, 100, 200, 400],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9]
        }
    elif fe_model_name == 'LGBMRegressor':
        param_grid = {
            'n_estimators': [50, 100, 200, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8, 10],
            'min_child_samples': [5, 10, 20],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8]
        }
    else:
        raise ValueError("Unknown regressor for hyperparameter tuning.")

    tuner = GridSearchCV(fe_model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1).fit(x, y)
    return tuner.best_estimator_, tuner.best_params_