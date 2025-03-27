import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, linregress, zscore
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, MinMaxScaler, OneHotEncoder

def encode_categorical(df_train: pd.DataFrame, df_test: pd.DataFrame, cat_var: list[str], reference=list[str]):
    """
    Encode categorical variables using one-hot encoding.
    """
    encoder = OneHotEncoder(drop=reference, sparse_output=False, dtype=int)
    encoded_train = encoder.fit_transform(df_train[cat_var])
    encoded_test = encoder.transform(df_test[cat_var])
    feature_names = encoder.get_feature_names_out([*cat_var])

    df_train_encoded = pd.DataFrame(encoded_train, columns=feature_names, index=df_train.index)
    df_test_encoded = pd.DataFrame(encoded_test, columns=feature_names, index=df_test.index)

    df_train = df_train.drop(cat_var, axis=1).join(df_train_encoded)
    df_test = df_test.drop(cat_var, axis=1).join(df_test_encoded)
    return df_train, df_test, encoder, encoder.get_feature_names_out().tolist()

def explore_numeric_vars(df, y_var: str, numeric_vars: list[str]):
    """
    Explore the distribution of numeric variables and the response variable.
    """
    print(df[[y_var] + numeric_vars].describe())
    mosaic_str = "t..;m..;crg"  # top, left, center, right, bottom
    for var in [y_var] + numeric_vars:
        mosaic = plt.figure(layout="constrained").subplot_mosaic(mosaic_str, height_ratios=[1, 1, 5], width_ratios=[5, 1, 1])
        mosaic['c'].scatter(df[var], df[y_var])
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
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def group_info(df: pd.DataFrame, group_var: str, min_size: int = None):
    """
    Display information about the groups in the DataFrame.
    """
    group_counts = df[group_var].value_counts()
    print(f"Unique {group_var}: {len(group_counts)}")
    print(f"Group sizes:\n{group_counts.describe()}")
    if min_size:
        valid_groups = group_counts[group_counts >= min_size].index
        df = df[df[group_var].isin(valid_groups)]
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
    df_clean = df.copy().dropna(subset=vars_to_clean)  # Work on a copy, drop NaNs
    
    if method == 'zscore':
        z_scores = np.abs(zscore(df_clean[vars_to_clean]))
        mask = (z_scores < zscore_threshold).all(axis=1)
    
    elif method == 'iqr':
        Q1 = df_clean[vars_to_clean].quantile(0.25)
        Q3 = df_clean[vars_to_clean].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        mask = ((df_clean[vars_to_clean] >= lower_bound) & 
                (df_clean[vars_to_clean] <= upper_bound)).all(axis=1)
    
    elif method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=lof_contamination)
        outlier_labels = lof.fit_predict(df_clean[vars_to_clean])
        mask = outlier_labels == 1  # 1 = inlier, -1 = outlier
    
    else:
        raise ValueError("Method must be 'zscore', 'iqr', or 'lof'")
    
    # Filter and reset index
    df_clean = df_clean[mask].reset_index(drop=True)
    print(f"Rows after outlier removal: {len(df_clean)} (Removed: {len(df) - len(df_clean)})")
    return df_clean

def plot_residuals(df, var, y_var, x_label, y_label):
    mosaic_str = "t.;cr"
    mosaic = plt.figure(layout="constrained").subplot_mosaic(mosaic_str, height_ratios=[1, 5], width_ratios=[5, 1])
    mosaic['c'].scatter(df[var], df[y_var])

    mosaic['t'].boxplot(df[var], orientation='horizontal', widths=0.7)
    mosaic['r'].boxplot(df[y_var], widths=0.7)

    sns.regplot(data=df, x=var, y=y_var, ci=95, line_kws={'color': 'red'}, ax=mosaic['c'])
    p_value = linregress(df[var], df[y_var])[3]
    mosaic['c'].text(0.95, 0.95, f'p-value: {p_value:.3f}', transform=mosaic['c'].transAxes, ha='right', va='top')

    mosaic['c'].set_xlabel(x_label)
    mosaic['c'].set_ylabel(y_label, rotation=0)
    for key in mosaic:
        if key != 'c':
            for spine in mosaic[key].spines.values():
                spine.set_visible(False)
            mosaic[key].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()

def shap_plots(model, X: np.ndarray, var: str):
    """
    explain the model's predictions using SHAP
    summarize the effects of all the features
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    shap.dependence_plot(var, shap_values, X)

def qq_plot(resid):
    """
    Normality of Residuals
    """
    fig, ax = plt.subplots(figsize=(5/2.54, 5/2.54), layout='constrained')
    sm.qqplot(resid, line='s', ax=ax)
    ax.set_title('Q-Q Plot of Residuals')
    plt.show()

def evaluate_metrics(y_obs, y_pred):
    metrics = {
        'MAE': mean_absolute_error(y_obs, y_pred),
        'MAPE': mean_absolute_percentage_error(y_obs, y_pred),
        'MSE': mean_squared_error(y_obs, y_pred),
        'R²': r2_score(y_obs, y_pred),
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

    plt.figure(figsize=(8/2.54, 8/2.54), layout='constrained')
    plt.scatter(y_tr_obs, y_tr_pred, color='tab:blue',
                label=f"Train (R² = {metric_tr['R²']:.3f}, r = {metric_tr['r']:.3f})")
    plt.scatter(y_te_obs, y_te_pred, color='tab:orange',
                label=f"Test (R² = {metric_te['R²']:.3f}, r = {metric_te['r']:.3f})")
    plt.plot([0, 1], [0, 1], color='tab:red', linestyle='--',
             label='Perfect Fit', transform=plt.gca().transAxes, zorder=3)
    plt.xlabel(f"Observed {var}")
    plt.ylabel(f"Predicted {var}")
    plt.legend(loc='upper left', frameon=False)
    plt.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.0, 0.95), ncol=1)
    plt.show()

def create_data(train_data, scalerx, encoderx, vary_param, vary_range, fixed_params):
    """
    Create evaluation dataframe with one varying parameter and others fixed.
    
    Parameters:
    - vary_param: str, parameter to vary
    - vary_range: array-like, range of values for the varying parameter
    - fixed_params: dict, fixed parameter values
    """
    df_eval = pd.DataFrame(columns=train_data.columns)
    df_eval[vary_param] = vary_range
    
    for param, value in fixed_params.items():
        df_eval[param] = value
    
    return df_eval, scalerx.transform(df_eval)

# def best_fe_model(x, y):
#         """
#         Tune hyperparameters of the fixed effect regressor
#         """
#         fe_model_name = type(fe_model).__name__
#         if fe_model_name == 'RandomForestRegressor':
#             from scipy.stats import randint, uniform
#             param_dist = {
#                 'n_estimators': randint(5, 400),              # Number of trees in the forest
#                 'max_depth': randint(2, 15),                  # Maximum depth of the trees
#                 'min_samples_split': randint(2, 18),          # Minimum number of samples required to split an internal node
#                 'min_samples_leaf': randint(1, 10),           # Minimum number of samples required to be at a leaf node
#                 'max_samples': uniform(0.5, 0.4)              # Fraction of samples to draw from x to train each base estimator
#             }
#         elif fe_model_name == 'MLPRegressor':
#             # param_dist = {
#             #     'hidden_layer_sizes': [(5,), (8,), (12,), (18,), (25,), (50,)]}
#             param_dist = {
#                 'hidden_layer_sizes': [(5,), (10,), (5, 5), (5, 5, 5), (100, 100), (50, 50, 50)],
#                 'activation': ['relu', 'tanh', 'logistic'],  # 'logistic' will be tested again here
#                 'solver': ['adam', 'sgd', 'lbfgs'],
#                 'learning_rate': ['constant', 'adaptive']}
#         elif fe_model_name == 'CatBoostRegressor':
#             param_dist = {
#                 'iterations': (5, 300),
#                 'learning_rate': (0.01, 0.5),
#                 'depth': (2, 15),
#                 'l2_leaf_reg': (1, 7),
#                 'bagging_temperature': (0.5, 1.5)}
#         elif fe_model_name == 'GradientBoostingRegressor':
#             param_dist = {
#                 'n_estimators': (5, 400),
#                 'learning_rate': (0.01, 0.5),
#                 'max_depth': (2, 15),
#                 'min_samples_split': (2, 10),
#                 'min_samples_leaf': (1, 8)}
#         elif fe_model_name == 'xGBRegressor':
#             param_dist = {
#                 'n_estimators': (5, 400),
#                 'max_depth': (2, 15),
#                 'learning_rate': (0.001, 0.1),
#                 'min_child_weight': (1, 5),
#                 'subsample': (0.5, 0.9),
#                 'colsample_bytree': (0.5, 0.9)}
#         elif fe_model_name == 'LGBMRegressor':
#             param_dist = {
#                 'n_estimators': (5, 400),
#                 'learning_rate': (0.001, 0.1),
#                 'max_depth': (2, 15),
#                 'min_child_samples': (5, 40),
#                 'subsample': (0.7, 0.9),
#                 'colsample_bytree': (0.6, 0.9)}
#         else:
#             raise ValueError("Unknown regressor for hyperparameter tuning.")
#         opt = RandomizedSearchCV(fe_model, param_dist, cv=5,
#                            scoring='neg_mean_squared_error', n_jobs=-1, n_iter=20).fit(x, y)
#         fe_model, fe_model_params = opt.best_estimator_, opt.best_params_
#         return 