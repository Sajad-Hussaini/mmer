from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import norm, lognorm, expon, gamma, weibull_min, beta, pearsonr, linregress
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as smm
from scipy import stats
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import cumulative_trapezoid as cumtrapz
import pandas as pd
from patsy import dmatrix
import sys
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.neighbors import LocalOutlierFactor
from tabulate import tabulate
sys.path.append('C:/Users/Sajad/OneDrive - Universidade do Minho/Python-Scripts/Python-Script-simulation')
# sys.path.append('C:/Users/future/OneDrive - Universidade do Minho/Python-Scripts/Python-Script-simulation')
import simmodel as sm
# =============================================================================
# TODO Step 1: Load and preprocess your data
# =============================================================================
plt.rcParams['figure.dpi'] = 600
df = pd.read_csv(sm.Tools().open_file()[0])

df = df.dropna(subset=['Mw', 'Rjb'])
# df = df[(df['PGA'] <= 1000) & (df['PGA'] >= 100)]
def clean_outliers(df, method='zscore', zscore_threshold=3,
                   lof_n_neighbors=10, lof_contamination=0.1):
    """
    Methods: zscore, iqr, lof
    """
    if method == 'zscore':
        # finiding outliers using z-scores
        z_scores = np.abs((df[['Mw', 'Rjb']] - df[['Mw', 'Rjb']].mean()) / df[['Mw', 'Rjb']].std())
        outlier_threshold = zscore_threshold
        outliers = df[(z_scores > outlier_threshold).any(axis=1)]
        df_cleaned = df.drop(outliers.index)

    elif method == 'iqr':
        # finding outliers using IQR (Interquartile Range)
        Q1 = df[['Mw', 'Rjb']].quantile(0.25)
        Q3 = df[['Mw', 'Rjb']].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = df[((df[['Mw', 'Rjb']] < lower_bound) | (df[['Mw', 'Rjb']] > upper_bound)).any(axis=1)]
        df_cleaned = df.drop(iqr_outliers.index)

    elif method == 'lof':
        # finding outliers using LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=lof_contamination)
        outlier_scores = lof.fit_predict(df[['Mw', 'Rjb']])
        lof_outliers = df[np.where(outlier_scores == -1, 'Outlier', 'Inlier') == 'Outlier']
        df_cleaned = df.drop(lof_outliers.index)

    else:
        print("Invalid method specified. Please choose from 'zscore', 'iqr', or 'lof'.")
        return None

    return df_cleaned

# cleaning process
# df = clean_outliers(df, method='iqr')
#%% Initial Feature selection
# Calculate correlation matrix for specific columns
correlation_matrix = df[['Mw', 'Rjb', 'PGA']].corr()
print(f"Correlation with PGA:\n{correlation_matrix['PGA']}")
#%%============================================================================
# TODO Step 2: Split data into features (X) and target (Y) with transformations
# =============================================================================
X = df[['Event', 'Mw', 'Rjb']]
Y = df[['PGA', 'PGV', 'PGD', *list(df.columns)[43:-1]]]
var = 'PGA'

def split_data(X, Y, rand_st=42, test_size=0.2, scalerx_method=None, scalery_method=None):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=rand_st)

    if scalerx_method:
        # Initialize X scaler
        scalerx = None
        if scalerx_method == 'MinMaxScaler':
            scalerx = MinMaxScaler(feature_range=(0.2, 0.8))
        elif scalerx_method == 'RobustScaler':
            scalerx = RobustScaler()
        else:
            print("Invalid X scaler method. Supported methods are 'MinMaxScaler' and 'RobustScaler'.")
            return None

        # Scale X data
        X_train[['Mw', 'Rjb']] = scalerx.fit_transform(X_train[['Mw', 'Rjb']])
        X_test[['Mw', 'Rjb']] = scalerx.transform(X_test[['Mw', 'Rjb']])

    if scalery_method:
        # Initialize Y scaler
        scalery = None
        if scalery_method == 'Boxcox':
            Y_train, lambda_y = boxcox(Y_train)
            Y_test = boxcox(Y_test, lmbda=lambda_y)
        elif scalery_method == 'PowerTransformer':
            scalery = PowerTransformer()
        elif scalery_method == 'RobustScaler':
            scalery = RobustScaler()
        elif scalery_method == 'Log':
            Y_train = np.log(Y_train)
            Y_test = np.log(Y_test)
        else:
            print("Invalid Y scaler method. Supported methods are 'Boxcox', 'PowerTransformer', 'RobustScaler', and 'LogTransform'.")
            return None

        # Transform Y data
        if scalery_method in ['PowerTransformer', 'RobustScaler']:
            Y_train = scalery.fit_transform(Y_train.reshape(-1, 1)).reshape(-1)
            Y_test = scalery.transform(Y_test.reshape(-1, 1)).reshape(-1)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = split_data(X, Y, rand_st=42, test_size=0.3,
                                              scalerx_method=None, scalery_method='Log')
# %% Visualize the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.scatter(df['Mw'], df[var], c='tab:blue', alpha=0.5)
ax1.set(xlabel='Mw', ylabel=var)

ax2.scatter(df['Rjb'], df[var], c='tab:blue', alpha=0.5)
ax2.set(xlabel='Rjb')
plt.tight_layout()
plt.show()
# transformed response variable
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.hist(Y_train[var], density=True, alpha=1, color='tab:blue')
ax1.set(title=f'Histogram of Transformed {var}', xlabel=f'{var}', ylabel='Density')

smm.qqplot(Y_train[var], line='s', ax=ax2)
ax2.set(title=f'Q-Q Plot of Transformed {var}', xlabel='Theoretical Quantiles',
        ylabel='Sample Quantiles')
plt.tight_layout()
plt.show()
#%%============================================================================
# TODO Fit a Linear Mixed-Effects Model (LMM)
# =============================================================================
# random intercept for group
lmm_model = smm.MixedLM.from_formula(formula=f"{var} ~ Mw + Rjb",
                                      data=X_train.join(Y_train), groups='Event')

model = lmm_model.fit()
# Between-event variance (single random effect)
be_var = model.cov_re.values[0][0]
# Within-event variance (scalar error variance)
we_var = model.scale
print(model.summary())
residuals = model.resid
X_train['we_residual'] = residuals

df_re = pd.DataFrame.from_dict(
    model.random_effects, orient='index').reset_index()
df_re.columns = ['Event', 'be_residual']
X_train = X_train.merge(df_re, on='Event', how='left')
X_train['t_residual'] = X_train['we_residual'] + X_train['be_residual']
# random_effect_values = pd.DataFrame(mdf.random_effects).T
            # random_effect_values.index.name = 'Group'
            # random_effect_values.columns = ['Value']

#%% Check some assumptions
# Residual plot for heteroscedasticity, checking for patterns in the residuals.
# RVF plot
sns.residplot(x=model.predict(), y=X_train['we_residual'], lowess=True,
              scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')  # Independence of Residuals
plt.show()
# Normality of Residuals (also you can plot histogram for residual)
smm.qqplot(X_train['we_residual'], line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()
# %%
# Plot a scatter plot between an independent variable and the dependent variable
# Linearity of Relationships
cols = ['Mw', 'Rjb']
res_type = ['be_residual', 'we_residual']
xname = [r'$M_w$', r'$R_{jb}\,(km)$']
yname = [r'$\eta$', r'$\epsilon$']
fig = plt.figure(figsize=(15, 5))  # Adjust the figure size as needed
gs = gridspec.GridSpec(1, 2)
for i in range(2):
    ax = plt.subplot(gs[0, i])
    # sns.residplot(x=X_train[cols[i]], y=X_train[res_type[i]], lowess=True,
                    # scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    sns.regplot(x=X_train[cols[i]], y=X_train[res_type[i]], ci=95, line_kws={'color': 'red'})
    plt.xlabel(f'{xname[i]}', rotation=0)
    plt.ylabel(f'{yname[i]}', rotation=0)
plt.tight_layout()
plt.show()
# %%
# Predict all output parameters together
pred_tr = model.predict()
pred_te = model.predict(X_test)
# Evaluate the model
mae_tr = mean_absolute_error(Y_train[var], pred_tr)
mse_tr = mean_squared_error(Y_train[var], pred_tr)
rmse_tr = np.sqrt(mse_tr)
r_squared_tr = r2_score(Y_train[var], pred_tr)

mae_te = mean_absolute_error(Y_test[var], pred_te)
mse_te = mean_squared_error(Y_test[var], pred_te)
rmse_te = np.sqrt(mse_te)
r_squared_te = r2_score(Y_test[var], pred_te)
table_data = [
    ["Metric", "Test", "Train"],
    ["Mean Absolute Error (MAE)", f"{mae_te:.3f}", f"{mae_tr:.3f}"],
    ["Mean Squared Error (MSE)", f"{mse_te:.3f}", f"{mse_tr:.3f}"],
    ["Root Mean Squared Error (RMSE)", f"{rmse_te:.3f}", f"{rmse_tr:.3f}"],
    ["R-squared (R²)", f"{r_squared_te:.3f}", f"{r_squared_tr:.3f}"]
]
print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
# %%
Mwr = np.arange(6.8, 7.7, 0.1)
Rjb = np.arange(10, int(np.max(df['Rjb'])), 0.5)
# Create DataFrames for Mw and Rjb combinations
X_mw = pd.DataFrame({'Mw': Mwr, 'Rjb': 7.5})
X2_mw = X_mw.copy()
X2_mw['Rjb'] = 20

X_rjb = pd.DataFrame({'Mw': 7.5, 'Rjb': Rjb})
X2_rjb = X_rjb.copy()
X2_rjb['Mw'] = 6.9

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Plot for Mw
axes[0].scatter(df['Mw'], df[var], c='tab:blue', alpha=0.5)
axes[0].plot(X_mw['Mw'], np.exp(model.predict(X_mw)), c='tab:orange', label=f"Rjb={X_mw['Rjb'][0]}")
axes[0].plot(X2_mw['Mw'], np.exp(model.predict(X2_mw)), c='tab:green', label=f"Rjb={X2_mw['Rjb'][0]}")
axes[0].set_xlabel('Mw')
axes[0].set_ylabel(f'{var}')
axes[0].legend(frameon=False)
axes[0].set_yscale('log')

# Plot for Rjb
axes[1].scatter(df['Rjb'], df[var], c='tab:blue', alpha=0.5)
axes[1].plot(X_rjb['Rjb'], np.exp(model.predict(X_rjb)), c='tab:orange', label=f"Mw={X_rjb['Mw'][0]}")
axes[1].plot(X2_rjb['Rjb'], np.exp(model.predict(X2_rjb)), c='tab:green', label=f"Mw={X2_rjb['Mw'][0]}")
axes[1].set_xlabel('Rjb (km)')
axes[1].set_ylabel(f'{var}')
axes[1].legend(frameon=False)
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()
