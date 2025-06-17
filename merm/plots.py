import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import probplot, linregress
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from matplotlib.ticker import MaxNLocator
from .style import style
from . import utils

_CM = 1 / 2.54  # cm to inches conversion factor

def plot_log_likeligood(model):
    with style():
        plt.figure(figsize=(7*_CM, 7*_CM))
        plt.plot(range(1, len(model.logL) + 1), model.logL, marker='o')
        plt.title("Log-Likelihood")
        plt.xlabel("Iteration")
        plt.ylabel("LogL")
        plt.grid(True, which='major', linewidth=0.15, linestyle='--')
        plt.text(0.95, 0.95, f'LogL = {model.logL[-1]:.4f}', transform=plt.gca().transAxes, va='top', ha='right')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show(block=False)

def plot_residual_covariance(model, corr=False):
    with style():
        data = utils.cov_to_corr(model.residuals_covariance) if corr else model.residuals_covariance
        dim = data.shape[0]
        plt.figure(figsize=((dim + 5)*_CM, (dim + 5)*_CM))
        labels = [f"R{m+1}" for m in range(dim)]
        sns.heatmap(data, annot=True, cmap='coolwarm', vmin=-1 if corr else None, vmax=1 if corr else None,
                    xticklabels= labels, yticklabels=labels)
        default_title = r"Residual Correlation ($\phi$)" if corr else r"Residual Covariance ($\phi$)"
        plt.title(default_title)
        plt.show()

def plot_random_effects_covariance(model, corr=False):
    with style():
        for k in range(model.num_groups):
            data = utils.cov_to_corr(model.random_effects_covariance[k]) if corr else model.random_effects_covariance[k]
            dim = data.shape[0]
            labels = [f"R{m+1}-{'I' if q == 0 else f'S{q}'}" for m in range(model.num_res) for q in range(model.n_effect[k])]
            plt.figure(figsize=((dim + 5)*_CM, (dim + 5)*_CM))
            sns.heatmap(data, annot=True, cmap='coolwarm', vmin=-1 if corr else None, vmax=1 if corr else None,
                        xticklabels=labels, yticklabels=labels)
            default_title = fr"Random Effects Correlation ($\random_effects_covariance$) for Group {k+1}" if corr else fr"Random Effects Covariance ($\random_effects_covariance$) for Group {k+1}"
            plt.title(default_title)
            plt.show()

def plot_residual_hist(residuals):
    with style():
        for m in range(residuals.shape[1]):
            plt.figure(figsize=(7*_CM, 7*_CM))
            plt.hist(residuals[:, m], bins='auto', edgecolor='black')
            plt.title(f"Response {m+1} Residuals")
            plt.xlabel("Residual")
            plt.ylabel("Frequency")
            plt.grid(True, which='major', linewidth=0.15, linestyle='--')
            plt.show()

def plot_residual_qq(residuals):
    with style():
        for m in range(residuals.shape[1]):
            plt.figure(figsize=(7*_CM, 7*_CM))
            probplot(residuals[:, m], dist="norm", plot=plt)
            plt.title(f"Response {m+1} Residuals")
            plt.grid(True, which='major', linewidth=0.15, linestyle='--')
            plt.show()

def plot_random_effect_hist(random_effects, num_res, n_effect, n_level):
    with style():
        for k, mu_k in random_effects.items():
            mu_k = mu_k.reshape(num_res, n_effect[k], n_level[k])
            for m in range(num_res):
                for j in range(n_effect[k]):
                    plt.figure(figsize=(7*_CM, 7*_CM))
                    plt.hist(mu_k[m, j, :], bins='auto', edgecolor='black')
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    plt.title(f"Group {k+1} Response {m+1} Random {effect_name}")
                    plt.xlabel("Random Effect")
                    plt.ylabel("Frequency")
                    plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                    plt.show()

def plot_random_effect_qq(random_effects, num_res, n_effect, n_level):
    with style():
        for k, mu_k in random_effects.items():
            mu_k = mu_k.reshape(num_res, n_effect[k], n_level[k])
            for m in range(num_res):
                for j in range(n_effect[k]):
                    plt.figure(figsize=(7*_CM, 7*_CM))
                    probplot(mu_k[m, j, :], dist="norm", plot=plt)
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    plt.title(f"Group {k+1} Response {m+1} Random {effect_name}")
                    plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                    plt.show()

def plot_residuals_vs_fitted(fitted_value, random_effects, residuasls, num_res):
    with style():
        for m in range(num_res):
            plt.figure(figsize=(7*_CM, 7*_CM))
            plt.scatter(fitted_value[:, m], residuasls[:, m], alpha=0.5, edgecolor='black')
            plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
            plt.title(f"Response {m+1}: Residuals vs Fitted")
            plt.xlabel("Fitted Values")
            plt.ylabel("Residuals")
            plt.grid(True, which='major', linewidth=0.15, linestyle='--')
            plt.show()

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
        p_value = linregress(df[xcol], df[ycol])[3]
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