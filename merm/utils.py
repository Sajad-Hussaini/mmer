import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, linregress, zscore
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

rcp = {
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 9,

    'lines.linewidth': 0.5,

    'axes.titlesize': 'medium',
    'axes.linewidth': 0.2,

    'xtick.major.width': 0.2,
    'ytick.major.width': 0.2,
    'xtick.minor.width': 0.15,
    'ytick.minor.width': 0.15,

    'legend.framealpha': 1.0,
    'legend.frameon': False,

    'figure.dpi': 900,
    'figure.figsize': (10/2.54, 8/2.54),
    'figure.constrained_layout.use': True,

    'patch.linewidth': 0.5,
    }

def plot_data(df, y_var: str, numeric_vars: list[str]):
    """
    Explore the distribution of numeric variables and the response variable.
    """
    print(df[[y_var] + numeric_vars].describe())
    mosaic_str = "t..;m..;crg"  # top, left, center, right, bottom
    for var in [y_var] + numeric_vars:
        with plt.rc_context(rc=rcp):
            mosaic = plt.figure(figsize=(10/2.54, 10/2.54)).subplot_mosaic(mosaic_str, height_ratios=[1, 1, 5], width_ratios=[5, 1, 1])
            mosaic['c'].scatter(df[var], df[y_var], fc='none', ec='tab:blue')
            mosaic['c'].set_xlabel(var)
            mosaic['c'].set_ylabel(y_var)

            mosaic['m'].hist(df[var], bins='auto')
            mosaic['t'].boxplot(df[var], orientation='horizontal', widths=0.7)
            mosaic['r'].hist(df[y_var], bins='auto', orientation='horizontal')
            mosaic['g'].boxplot(df[y_var], widths=0.7)

            skew_value = df[var].skew().round(2)
            pos = {True: (0.05, 'left'), False: (0.95, 'right')}
            x_pos, ha = pos[skew_value < 0]
            mosaic['c'].text(x_pos, 0.95, f'Skew: {skew_value}', transform=mosaic['c'].transAxes, ha=ha, va='top')
            
            for key in mosaic:
                if key != 'c':
                    for spine in mosaic[key].spines.values():
                        spine.set_visible(False)
                    mosaic[key].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.show()
    corr_matrix = df[numeric_vars].corr()
    with plt.rc_context(rc=rcp):
        plt.figure(figsize=(5/2.54, 5/2.54))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()

def group_info(df: pd.DataFrame, group_var: str, min_size: int = None):
    """
    Display information about the groups in the DataFrame.
    """
    group_counts = df[group_var].value_counts()
    print(f"Unique {group_var}: {len(group_counts)}")
    print(f"Group sizes:\n{group_counts.describe()}")
    if min_size is not None:
        df = df.groupby(group_var).filter(lambda x: len(x) >= min_size)
        print(f"Remaining groups: {df[group_var].nunique()}")
    return df

def clean_outliers(df: pd.DataFrame, vars_to_clean: list[str], method: str = 'zscore',
                   zscore_threshold: float = 3, iqr_factor: float = 1.5,
                   lof_n_neighbors: int = 20, lof_contamination: float = 0.1) -> pd.DataFrame:
    """
    Remove outliers from specified variables in a DataFrame using the chosen method.

    Args:
        df: Input DataFrame
        vars_to_clean: List of column names to clean outliers from
        method: Outlier detection method ('zscore', 'iqr', or 'lof')
        zscore_threshold: Threshold for z-score method (default: 3)
        iqr_factor: Multiplier for IQR method (default: 1.5)
        lof_n_neighbors: Number of neighbors for LOF method (default: 20)
        lof_contamination: Expected proportion of outliers for LOF (default: 0.1)

    Returns:
        DataFrame with outliers removed
    """
    if method == 'zscore':
        z_scores = np.abs(zscore(df[vars_to_clean]))
        mask = (z_scores < zscore_threshold).all(axis=1)
    
    elif method == 'iqr':
        Q1 = df[vars_to_clean].quantile(0.25)
        Q3 = df[vars_to_clean].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        mask = ((df[vars_to_clean] >= lower_bound) & 
                (df[vars_to_clean] <= upper_bound)).all(axis=1)
    
    elif method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=lof_contamination)
        outlier_labels = lof.fit_predict(df[vars_to_clean])
        mask = outlier_labels == 1  # 1 = inlier, -1 = outlier
    
    else:
        raise ValueError("Method must be 'zscore', 'iqr', or 'lof'")
    
    df = df[mask].reset_index(drop=True)
    print(f"Rows after outlier removal: {len(df)}")
    return df

def plot_residuals(x, y, x_label, y_label):
    mosaic_str = "t.;cr"
    with plt.rc_context(rc=rcp):
        mosaic = plt.figure(figsize=(8/2.54, 8/2.54)).subplot_mosaic(mosaic_str, height_ratios=[1, 5], width_ratios=[5, 1])
        mosaic['c'].scatter(x, y, fc='none', ec='tab:blue')

        mosaic['t'].boxplot(x, orientation='horizontal', widths=0.7)
        mosaic['r'].boxplot(y, widths=0.7)

        sns.regplot(x=x, y=y, ci=95, line_kws={'color': 'red'}, ax=mosaic['c'])
        p_value = linregress(x, y)[3]
        mosaic['c'].text(0.95, 0.95, f'p-value: {p_value:.3f}', transform=mosaic['c'].transAxes, ha='right', va='top')

        mosaic['c'].set_xlabel(x_label)
        mosaic['c'].set_ylabel(y_label, rotation=0)
        for key in mosaic:
            if key != 'c':
                for spine in mosaic[key].spines.values():
                    spine.set_visible(False)
                mosaic[key].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
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
    with plt.rc_context(rc=rcp):
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
        'r': pearsonr(y_obs, y_pred)[0]}
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    return metrics

def print_metrics(metrics_tr, metrics_te):
    print(f"{'Metric':^10}{'Train':^10}{'Test':^10}")
    for metric, tr_value in metrics_tr.items():
        te_value = metrics_te.get(metric, None)
        print(f"{metric:^10}{tr_value:^10.3f}{te_value:^10.3f}")

def model_performance(y_tr_obs, y_tr_pred, y_te_obs, y_te_pred, var):
    metric_tr = evaluate_metrics(y_tr_obs, y_tr_pred)
    metric_te = evaluate_metrics(y_te_obs, y_te_pred)
    print_metrics(metric_tr, metric_te)

    with plt.rc_context(rc=rcp):
        plt.figure(figsize=(8/2.54, 8/2.54))
        plt.scatter(y_tr_obs, y_tr_pred, fc='none', ec='tab:blue',
                    label=fr"Train ($R^2$ = {metric_tr['r2']:.3f}, r = {metric_tr['r']:.3f})")
        plt.scatter(y_te_obs, y_te_pred, fc='none', ec='tab:orange',
                    label=f"Test ($R^2$ = {metric_te['r2']:.3f}, r = {metric_te['r']:.3f})")
        plt.plot([0, 1], [0, 1], color='tab:red', linestyle='--',
                label='Perfect Fit', transform=plt.gca().transAxes, zorder=3)
        plt.xlabel(f"Observed {var}")
        plt.ylabel(f"Predicted {var}")
        plt.legend(loc='upper left')
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

rcp = {
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'mathtext.default': 'regular',  # can be 'regular' 'it' etc
    'font.size': 9,

    'lines.linewidth': 0.5,
    'lines.markersize': 3,

    'boxplot.boxprops.linewidth': 0.5,
    'boxplot.whiskerprops.linewidth': 0.5,
    'boxplot.capprops.linewidth': 0.5,
    'boxplot.flierprops.markersize': 3,
    'boxplot.flierprops.markeredgewidth': 0.5,

    'axes.titlesize': 'medium',
    'axes.linewidth': 0.2,

    'xtick.major.width': 0.2,
    'ytick.major.width': 0.2,
    'xtick.minor.width': 0.15,
    'ytick.minor.width': 0.15,

    'legend.framealpha': 1.0,
    'legend.frameon': False,

    'figure.dpi': 900,
    'figure.figsize': (10/2.54, 8/2.54),
    'figure.constrained_layout.use': True,

    'patch.linewidth': 0.5,
    }