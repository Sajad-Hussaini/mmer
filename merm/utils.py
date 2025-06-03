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

def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)

def plot_pair(df: pd.DataFrame, xcol: str, ycol: str, x_label: str = None, y_label: str = None):
    " Plot scatter, distribution and boxplot of x and y "
    mosaic_layout = [['boxx', '.', '.'], ['distx', '.', '.'], ['scatter', 'disty', 'boxy']]
    x_label = xcol if x_label is None else x_label
    y_label = ycol if y_label is None else y_label
    with style():
        fig, axes = plt.subplot_mosaic(mosaic_layout, height_ratios=[1, 2, 8], width_ratios=[8, 2, 1])
        axes['scatter'].scatter(df[xcol], df[ycol], fc='none', ec='tab:blue')
        axes['scatter'].set_xlabel(x_label)
        axes['scatter'].set_ylabel(y_label)

        axes['distx'].hist(df[xcol], bins='auto')
        axes['boxx'].boxplot(df[xcol], orientation='horizontal', widths=0.7)

        axes['disty'].hist(df[ycol], bins='auto', orientation='horizontal')
        axes['boxy'].boxplot(df[ycol], widths=0.7)

        skewness = df[ycol].skew().round(2)
        pos = {True: (0.05, 'left'), False: (0.95, 'right')}
        x_pos, ha = pos[skewness < 0]
        axes['scatter'].text(x_pos, 0.95, f'Skewnes={skewness}', transform=axes['scatter'].transAxes, ha=ha, va='top')
        for key, ax in axes.items():
            if key != 'scatter':
                ax.set_frame_on(False)  # Hides all spines
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.show()

def plot_corr(df: pd.DataFrame, numeric_vars: list[str]):
    " Plot correlation matrix of numeric variables. "
    corr_matrix = df[numeric_vars].corr()
    with style():
        plt.figure(figsize=(18/2.54, 12/2.54))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()

def filter_group(df: pd.DataFrame, group: str, min_size: int = None):
    " Filter groups based on a minimum member size. "
    group_counts = df[group].value_counts()
    print(group_counts.describe())
    if min_size is not None:
        df = df.groupby(group).filter(lambda x: len(x) >= min_size)
        print(f"Remaining groups: {df[group].nunique()}")
    return df

def clean_outliers(df: pd.DataFrame, vars_to_clean: list[str], method: str= "iqr",
                   zscore_threshold: float = 3, iqr_factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified variables in a DataFrame using the chosen method.
    Parameters:
        df: Input DataFrame
        vars_to_clean: List of column names to clean outliers from
        method: Outlier detection method
            'zscore': Z-score method, assumes normal distribution, sensitive to outliers
            'iqr': Interquartile Range, good for skewed distributions, robust to non-normal data
        zscore_threshold: Threshold for z-score method (default: 3)
        iqr_factor: Multiplier for IQR method (default: 1.5)
    Returns:
        DataFrame with outliers removed
    """
    if method == 'zscore':
        z_scores = np.abs((df[vars_to_clean] - df[vars_to_clean].mean()) / df[vars_to_clean].std())
        mask = (z_scores < zscore_threshold).all(axis=1)

    elif method == 'iqr':
        Q1 = df[vars_to_clean].quantile(0.25)
        Q3 = df[vars_to_clean].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        mask = ((df[vars_to_clean] >= lower_bound) & (df[vars_to_clean] <= upper_bound)).all(axis=1)
    else:
        raise ValueError("Method must be 'zscore', 'iqr'")
    
    df_cleaned = df[mask].reset_index(drop=True)
    print(f"Rows after outlier removal: {df_cleaned.shape[0]}")
    return df_cleaned

def plot_residuals(df: pd.DataFrame, xcol: str, ycol: str, x_label: str = None, y_label: str = None):
    mosaic_layout = "t.;cr"
    x_label = xcol if x_label is None else x_label
    y_label = ycol if y_label is None else y_label
    with style():
        fig, axes = plt.subplot_mosaic(mosaic_layout, height_ratios=[1, 8], width_ratios=[8, 1])
        axes['c'].scatter(df[xcol], df[ycol], fc='none', ec='tab:blue')
        axes['t'].boxplot(df[xcol], orientation='horizontal', widths=0.7)
        axes['r'].boxplot(df[ycol], widths=0.7)

        sns.regplot(data=df, x=xcol, y=ycol, ci=95, line_kws={'color': 'red'}, ax=axes['c'])
        p_value = stats.linregress(df[xcol], df[ycol])[3]
        axes['c'].text(0.95, 0.95, f'p-value={p_value:.3f}', transform=axes['c'].transAxes, ha='right', va='top')
        axes['c'].set_xlabel(x_label)
        axes['c'].set_ylabel(y_label, rotation=0)
        for key, ax in axes.items():
            if key != 'c':
                ax.set_frame_on(False)  # Hides all spines
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.show()

def shap_plots(model, x: np.ndarray, var: str):
    """
    explain the model's predictions using SHAP
    summarize the effects of all the features
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    shap.summary_plot(shap_values, x)
    shap.summary_plot(shap_values, x, plot_type="bar")
    shap.dependence_plot(var, shap_values, x)

def qq_plot(resid):
    """
    Normality of Residuals
    """
    with style():
        plt.figure(figsize=(5/2.54, 5/2.54), layout='constrained')
        sm.qqplot(resid, line='s')
        plt.title('Q-Q Plot of Residuals')
        plt.show()

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

def generate_merm_data(n=1000, M=3, K=2, o_k=[50, 40], p=5, slope_columns=[[0], [0, 2]], seed=42):
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
    np.random.seed(seed)
    X = np.random.randn(n, p)
    groups = np.zeros((n, K), dtype=int)
    for k in range(K):
        groups[:, k] = np.repeat(np.arange(o_k[k]), n // o_k[k])

    rscovariate = [None] * K
    for k in range(K):
        if slope_columns[k] is not None:
            rscovariate[k] = X[:, slope_columns[k]]

    # Generate residual covariance (phi: M x M)
    true_phi = np.eye(M) * 0.5
    true_phi[np.triu_indices(M, 1)] = 0.2
    true_phi = true_phi + true_phi.T - np.eye(M) * 0.5

    # Generate random effects covariances (tau_k: M*q_k x M*q_k)
    true_tau = []
    q_k = [1 if slope_columns[i] is None else 1 + len(slope_columns[i]) for i in range(K)]
    # Example: set different diagonal/off-diagonal values for each k
    diag_vals = [0.4 + 0.2 * k for k in range(K)]      # e.g., 0.4, 0.6, 0.8, ...
    off_diag_vals = [0.1 + 0.05 * k for k in range(K)] # e.g., 0.1, 0.15, 0.2, ...
    for k in range(K):
        tau_k = np.eye(M * q_k[k]) * diag_vals[k]
        tau_k[np.triu_indices(M * q_k[k], 1)] = off_diag_vals[k]
        tau_k = tau_k + tau_k.T - np.eye(M * q_k[k]) * diag_vals[k]
        true_tau.append(tau_k)

    # Generate fixed effects
    true_cof = np.random.randn(p, M) * 0.5
    fX = X @ true_cof

    # eps = np.random.multivariate_normal(np.zeros(M), true_phi, size=n)
    eps = np.random.multivariate_normal(np.zeros(M * n), sparse.kron(true_phi, np.eye(n)).toarray()).reshape((n, M), order='F')

    rand_eff = np.zeros((n, M))
    for k in range(K):
        Z_k, q_k_val, o_k_val = design_Z(groups[:, k], X[:, slope_columns[k]] if (slope_columns is not None and slope_columns[k] is not None) else None)
        IM_Z_k = sparse.kron(sparse.eye(M), Z_k)
        b_k = np.random.multivariate_normal(np.zeros(M * q_k[k] * o_k[k]), sparse.kron(true_tau[k], np.eye(o_k[k])).toarray())
        rand_eff += (IM_Z_k @ b_k).reshape((n, M), order='F')

    Y = fX + rand_eff + eps
    return X, Y, groups, slope_columns, true_phi, true_tau, true_cof

def design_Z(group: np.ndarray, random_slope_covariates: np.ndarray = None):
        """
        Construct random effects design matrix for a grouping factor.
        
        Parameters:
            group: (n_samples,) array of group levels.
            random_slope_covariates: (n_samples, q) array for random slopes (optional).
        
        Returns:
            Z_k: Sparse matrix (n_samples, o_k * q_k).
            q_k: Number of random effects per level.
            o_k: Number of unique levels.
        """
        levels, level_indices = np.unique(group, return_inverse=True)
        n = group.shape[0]
        o = len(levels)
        q = 1 if random_slope_covariates is None else 1 + random_slope_covariates.shape[1]
        # Number of non-zero elements
        nnz = n * q
        rows = np.zeros(nnz, dtype=int)
        cols = np.zeros(nnz, dtype=int)
        data = np.zeros(nnz, dtype=float)
        for i in range(n):
            j = level_indices[i]  # level index (0 to o-1)
            base_idx = i * q      # Starting (intercept) index in sparse arrays
            # Intercept
            rows[base_idx] = i
            cols[base_idx] = j    # 0 * o + j
            data[base_idx] = 1.0
            # Slopes
            if random_slope_covariates is not None:
                for rs_idx in range(random_slope_covariates.shape[1]):
                    idx = base_idx + (rs_idx + 1)
                    rows[idx] = i
                    cols[idx] = (rs_idx + 1) * o + j
                    data[idx] = random_slope_covariates[i, rs_idx]
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, q * o)), q, o