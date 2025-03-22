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


def config_plot(dpi=600, font='Times New Roman', lw=1, marker_size=1, fontsize=9,
                ax_lbsize=9, legend_fsize=9, axlw = 0.25):
    plt.rcParams.update({
        'figure.dpi': dpi,
        'lines.linewidth': lw,
        'lines.markersize': marker_size,
        'font.size': fontsize,
        'axes.labelsize': ax_lbsize,
        'legend.fontsize': legend_fsize,
        'font.family': font,
        'axes.linewidth': axlw,
        'xtick.major.width': axlw,
        'xtick.minor.width': axlw,
        'ytick.major.width': axlw,
        'ytick.minor.width': axlw})
    cm = 1/2.54  # centimeters in inches
    return cm

def clean_outliers(data: pd.DataFrame, var_to_clean: list[str], method='zscore',
                   zscore_threshold=3, lof_n_neighbors=10, lof_contamination=0.1):
    """
    Clean outliers from the specified variable(s) in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        var_to_clean (list): The variable(s) to clean outliers from.
        method (str): The method to use for outlier detection ('zscore', 'iqr', or 'lof').
        zscore_threshold (float): The z-score threshold for the z-score method.
        lof_n_neighbors (int): The number of neighbors for the LOF method.
        lof_contamination (float): The contamination parameter for the LOF method.

    Returns:
        DataFrame: The cleaned DataFrame.
    """
    df = data.copy()
    df = df.dropna(subset=var_to_clean)
    if method == 'zscore':
        z_scores = np.abs(zscore(df[var_to_clean]))
        outliers = df[(z_scores > zscore_threshold).any(axis=1)]
    elif method == 'iqr':
        q1, q3 = np.percentile(df[var_to_clean], [25, 75], axis=0)
        iqr_value = q3 - q1
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value
        outliers = df[((df[var_to_clean] < lower_bound) | (
            df[var_to_clean] > upper_bound)).any(axis=1)]
    elif method == 'lof':
        lof = LocalOutlierFactor(
            n_neighbors=lof_n_neighbors, contamination=lof_contamination)
        outlier_scores = lof.fit_predict(df[var_to_clean])
        outliers = df[outlier_scores == -1]
    else:
        print("Invalid method specified. Please choose from 'zscore', 'iqr', or 'lof'.")
        return None
    return df.drop(outliers.index).reset_index(drop=True)

def plot_joint_grid(data, x_col, y_col, x_label, y_label):
    cm = config_plot(lw=0.5)
    g = sns.JointGrid(data=data, x=x_col, y=y_col, height=8*cm, ratio=8, space=0.01)
    sns.boxplot(data=data, x=x_col, ax=g.ax_marg_x, flierprops={"marker": "D"})
    sns.boxplot(data=data, y=y_col, ax=g.ax_marg_y, flierprops={"marker": "D"})
    sns.regplot(data=data, x=x_col, y=y_col, ci=95, line_kws={'color': 'red'}, ax=g.ax_joint)
    p_value = linregress(data[x_col], data[y_col])[3]
    g.ax_joint.text(0.95, 0.95, f'p-value: {p_value:.3f}', transform=g.ax_joint.transAxes, ha='right', va='top')
    g.set_axis_labels(x_label, y_label, rotation=0)
    plt.show()

def plot_res(data, x_col, y_col, x_label, y_label, plotname, fig_size):
    plt.figure(figsize=fig_size)
    sns.regplot(data=data, x=x_col, y=y_col, ci=95, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})
    # plt.scatter(data[x_col], data[y_col], alpha=0.5)
    # slope, intercept, r_value, p_value, std_err = linregress(data[x_col], data[y_col])
    # regression_line = slope * data[x_col] + intercept
    # plt.plot(data[x_col], regression_line, color='red')
    # p_value = linregress(data[x_col], data[y_col])[3]
    # plt.text(0.5, 0.1, f'p-value: {p_value:.3f}', transform=plt.gca().transAxes, ha='center', va='top')
    plt.text(0.5, 0.90, r'$\mathregular{ln(PGA) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "PGA" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PGV) (\frac{cm}{s})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "PGV" in plotname else None
    plt.text(0.5, 0.90,  r'$\mathregular{ln(PSA_{T=0.2s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa0_2" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PSA_{T=0.5s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa0_5" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PSA_{T=1s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa1" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PSA_{T=2s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa2" in plotname else None
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([-1.5, 1.5])
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(rf'C:\Users\Sajad\OneDrive - Universidade do Minho\Record-Database\ESM-database-Europe-SJ\streamlit-Tabriz-GMM\Paper\{plotname}.png', dpi=600)
    # plt.show()

def plot_res6(data, x_col, y_col, x_label, y_label, plotname, fig_size):
    plt.figure(figsize=fig_size)
    sns.regplot(data=data, x=x_col, y=y_col, ci=95, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})
    # plt.scatter(data[x_col], data[y_col], alpha=0.5)
    # slope, intercept, r_value, p_value, std_err = linregress(data[x_col], data[y_col])
    # regression_line = slope * data[x_col] + intercept
    # plt.plot(data[x_col], regression_line, color='red')
    # p_value = linregress(data[x_col], data[y_col])[3]
    # plt.text(0.5, 0.1, f'p-value: {p_value:.3f}', transform=plt.gca().transAxes, ha='center', va='top')
    plt.text(0.5, 0.90, r'$\mathregular{ln(PGA) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "PGA" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PGV) (\frac{cm}{s})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "PGV" in plotname else None
    plt.text(0.5, 0.90,  r'$\mathregular{ln(PSA_{T=0.2s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa0_2" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PSA_{T=0.5s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa0_5" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PSA_{T=1s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa1" in plotname else None
    plt.text(0.5, 0.90, r'$\mathregular{ln(PSA_{T=2s}) (\frac{cm}{s^2})}$', transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if "Sa2" in plotname else None
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([-1.5, 1.5])
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(rf'C:\Users\Sajad\OneDrive - Universidade do Minho\Record-Database\ESM-database-Europe-SJ\streamlit-Tabriz-GMM\Paper\{plotname}.png', dpi=600)
    # plt.show()

def variance_plot(data):
    cm = config_plot(lw=0.5, marker_size=3)
    plt.figure(figsize=(12*cm, 5*cm))
    # list_name = [r'$\mathregular{ln(PGA) (cm/s^2)}$',
                 # r'$\mathregular{ln(PGV) (cm/s)}$']
    # list_name.extend([rf'$\mathregular{{ln(PSA_{{T={t:.1f}s}}) (cm/s^2)}}$'
                      # for t in [0.2, 0.5, 1, 2]])
    # list_name.extend([rf'$\mathregular{{ln(PSA_{{T={t:.2f}s}}) (\frac{{cm}}{{s^2}})}}$'
                      # for t in np.arange(0.05, 2.05, 0.05)])
    list_name = data.index
    for (index, row), label_name in zip(data.iterrows(), list_name):
        if label_name == 'dP2':
            label_name = 'P2'
        if label_name == 'dC1':
            label_name = 'C1'
        plt.scatter(label_name, row['std_between'], color='tab:blue', marker='o')
        plt.scatter(label_name, row['std_within'], color='tab:orange', marker='s')
        plt.scatter(label_name, row['std_total'], color='tab:green', marker='^')
    plt.legend([r'$\mathregular{\tau}$', r'$\mathregular{\sigma}$', r'$\mathregular{\phi}$'],
               ncol=3, loc='lower left', bbox_to_anchor=(0.0, 0.97))
    plt.ylabel('StD of the residual', rotation=90)
    plt.xticks(rotation=30)
    # plt.ylim([0, 0.4])
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.gca().tick_params(axis='x', which='minor', bottom=False)
    plt.tight_layout()
    plt.show()

def metric_plot(data, data2):
    cm = config_plot(lw=0.5, marker_size=3)
    plt.figure(figsize=(12*cm, 5*cm))
    # list_name = [r'$\mathregular{ln(PGA) (cm/s^2)}$',
                 # r'$\mathregular{ln(PGV) (cm/s)}$']
    # list_name.extend([rf'$\mathregular{{ln(PSA_{{T={t:.1f}s}}) (cm/s^2)}}$'
                      # for t in [0.2, 0.5, 1, 2]])
    # list_name.extend([rf'$\mathregular{{ln(PSA_{{T={t:.2f}s}}) (\frac{{cm}}{{s^2}})}}$'
                      # for t in np.arange(0.05, 2.05, 0.05)])
    list_name = data.index
    for (index, row), label_name in zip(data.iterrows(), list_name):
        if label_name == 'dP2':
            label_name = 'P2'
        if label_name == 'dC1':
            label_name = 'C1'
        sc1 = plt.scatter(label_name, row['R²'], color='tab:blue', marker='o')
        sc2 = plt.scatter(label_name, row['r'], color='tab:orange', marker='s')
        sc3 = plt.scatter(label_name, row['RMSE'], color='tab:green', marker='D')
        sc4 = plt.scatter(label_name, row['MAPE'], color='tab:red', marker='^')
    # for (index, row), label_name in zip(data2.iterrows(), list_name):
    #     sc5 = plt.scatter(label_name, row['R²'], color='tab:blue', marker='o', alpha = 0.5, facecolors='none')
    #     sc6 = plt.scatter(label_name, row['r'], color='tab:orange', marker='s', alpha = 0.5, facecolors='none')
    #     sc7 = plt.scatter(label_name, row['RMSE'], color='tab:green', marker='D', alpha = 0.5, facecolors='none')
    #     sc8 = plt.scatter(label_name, row['MAPE'], color='tab:red', marker='^', alpha = 0.5, facecolors='none')
    # plt.legend([sc1, sc5, sc2, sc6, sc3, sc7, sc4, sc8],[
    #     r'$\mathregular{R^2_{test}}$', r'$\mathregular{R^2_{train}}$',
    #     r'$\mathregular{r_{test}}$', r'$\mathregular{r_{train}}$',
    #     r'$\mathregular{RMSE_{test}}$', r'$\mathregular{RMSE_{train}}$',
    #     r'$\mathregular{MAPE_{test}}$', r'$\mathregular{MAPE_{train}}$'],
    #            loc='lower left', ncol=4, bbox_to_anchor=(0, 1), fancybox=True, shadow=True)
    plt.legend([sc1, sc2, sc3, sc4],[
        r'$\mathregular{R^2}$',
        r'$\mathregular{r}$',
        r'$\mathregular{RMSE}$',
        r'$\mathregular{MAPE}$'],
               loc='lower left', ncol=4, bbox_to_anchor=(0, 1), fancybox=True, shadow=True)
    plt.ylabel('Performance metrics', rotation=90)
    # plt.xticks(rotation=30)
    plt.xticks(rotation=30)
    # plt.ylim([0, 2])
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.gca().tick_params(axis='x', which='minor', bottom=False)
    plt.tight_layout()
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

def variable_plot(df, x_var: list, y_var: str):
    for var in x_var:
        plt.scatter(df[var], df[y_var], c='tab:blue', alpha=0.5)
        plt.xlabel(var)
        plt.ylabel(y_var)
        plt.tight_layout()
        plt.show()

def qq_plot(resid):
    """
    Normality of Residuals
    """
    cm = config_plot(lw=0.5)
    fig, ax = plt.subplots(figsize=(5*cm, 5*cm), layout='constrained')
    sm.qqplot(resid, line='s', ax=ax)
    ax.set_title('Q-Q Plot of Residuals')
    plt.show()

def evaluate_metrics_df(y_obs, y_pred, y):
    metrics = {
        'MAE': mean_absolute_error(y_obs, y_pred),
        'MAPE': mean_absolute_percentage_error(y_obs, y_pred),
        'MSE': mean_squared_error(y_obs, y_pred),
        'R²': r2_score(y_obs, y_pred),
        'r': pearsonr(y_obs, y_pred)[0]}
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    return pd.DataFrame(metrics, index=[y])

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

    cm = config_plot(lw=0.5)
    plt.figure(figsize=(8*cm, 8*cm), layout='constrained')
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

def _plot_scatter(fig: plt.figure, gs: gridspec, data: pd.DataFrame, x: str,
                  y: str, xlabel: str, ylabel: str, text: str):
    ax = fig.add_subplot(gs)
    sns.regplot(data=data, x=x, y=y, ax=ax, ci=95, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3, 's': 2})
    _configure_subplot(fig, ax, xlabel, ylabel, text)

def _plot_line(fig: plt.figure, gs: gridspec, data: pd.DataFrame, x: str,
                  y: str, xlabel: str, ylabel: str, text: str):
    ax = fig.add_subplot(gs)
    ax.plot(data=data, x=x, y=y, ax=ax, ci=95, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})
    _configure_subplot(fig, ax, ylabel, xlabel, text)

def _configure_subplot(fig: plt.figure, ax: plt.axes, xlabel: str, ylabel: str, text: str):
    # ax.grid(False, linestyle='--', which='both')
    ax.minorticks_on()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylabel is None:
        ax.set_yticklabels([])
    if xlabel is None:
        ax.set_xticklabels([])
    ax.set_ylim(-1, 1)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(10))
    fig.text(0.5, 0.90, text, transform=ax.transAxes, ha='center', va='top',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round')) if text else None

def plot_residual(data, y_values: list[str], x_labels: str, y_labels: str, text: str):
    cm = config_plot(lw=0.5)
    fig = plt.figure(figsize=(12*cm, 5*cm), layout='constrained')
    gs = gridspec.GridSpec(3, 2)
    for idx, y_var in enumerate(y_values):
        data = pd.read_csv('resid_df.csv')
        if idx // 2 in [0, 1]:
           x_label = None
           y_label = y_labels[1] if idx % 2 == 0 else None
        elif idx // 2 == 2:
            x_label = x_labels[1]
            y_label = y_labels[1] if idx % 2 == 0 else None
        _plot_scatter(fig, gs[idx // 2, idx % 2], data, 'rjb', f'res_within_{y_var}', x_label, y_label, text[idx])
    plt.show()

def create_data(train_data, scalerx, vary_param, vary_range, fixed_params):
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