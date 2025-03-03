from MEML import MEML
import Funcs_gmm as fm
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import sys
sys.path.append(r'C:\Users\Future\OneDrive - Universidade do Minho\Python-Scripts\Python-Script-simulation')
import simmodel as sm
import joblib


if True:
    plt.rcParams["font.family"] = "Palatino Linotype"
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['font.size'] = 9
    # plt.rcParams['lines.markersize'] = 5
    cm = 1/2.54  # centimeters in inches
# TODO Step 1: Load, Preprocess, and Split the data
df_main = sm.Tools().csv_reader()[0]
X = ['Mw', 'Rjb', 'Focal depth']; cluster = 'Event'; re = 'RE'; y = 'PGA'
df = fm.clean_outliers(df_main, [*X, y], method='zscore')
df['RE'] = 1  # add a random intercep for random effect
data = df[[*X, cluster, re, y]]
train_data, test_data = train_test_split(df, test_size=0.20, random_state=42)

# Visualize the data
# fm.variable_plot(df, x_var=X, y_var=y)
# %% Mixed effect Model
# transforming the y values should be considered when non-normality and/or unequal variances are the main problems with the model
start = time.perf_counter()
model = MEML(MLPRegressor((5,), "logistic", random_state=42, solver='lbfgs'),
             RobustScaler(), np.log, max_iter=3, gll_stop=0.001, tuning=False)
model.fit(data=train_data, X=X, cluster=cluster, Z=re, y=y, data_val=test_data)
end = time.perf_counter()
print(f'All tasks finished in {end-start}')
# %% Further model investigations
model.training_graph()
var_within = model.var_within
var_between = model.var_between
fitval = model.predict(train_data, X, cluster, re)
res_between = model.res_between
res_within = model.res_within
res_total = res_within + res_between
# %% explain the model's predictions using SHAP
# fm.shap_plots(model.trained_fe_model, X_tr, var='Mw')
# %% Check assumptions heteroscedasticity, patterns in residuals vs variables or fitval
if False:
    fm.qq_plot(res_within)
    var_cols = X
    res_cols = ['res_between', 'res_within', 'res_between', 'res_within']
    xlabel = [r'$M_w$', r'$R_{jb}\,(km)$', r'$FD\,(km)$',
              'fitted value', r'$V_{s30}\,(\frac{m}{s})$']
    ylabel = [r'$\eta$', r'$\epsilon$', r'$\eta$', 'residual']
    for i in range(4):
        fm.plot_joint_grid(
            train_data, var_cols[i], res_cols[i], xlabel[i], ylabel[i])
# %% Model Performances
if False:
    train_pred = model.predict(train_data, X, cluster, re)  # fitval
    test_pred = model.predict(test_data, X, cluster, re)
    fm.model_performance(np.log(train_data[y]), train_pred, np.log(test_data[y]), test_pred, y)
# %% Plot attenuation result
if True:
    cols = [*X, cluster, re]

    X1 = fm.create_data_mw(cols, np.arange(6.8, 7.7, 0.1), 10, 14)
    X2 = fm.create_data_mw(cols, np.arange(6.8, 7.7, 0.1), 20, 14)
    X3 = fm.create_data_rjb(cols, np.arange(10, 85, 0.5), 7.5, 14)
    X4 = fm.create_data_rjb(cols, np.arange(10, 85, 0.5), 7, 14)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot for Mw
    axes[0].scatter(df['Mw'], df[y], c='tab:blue', alpha=0.5)
    axes[0].plot(X1['Mw'], np.exp(model.predict(X1, X, cluster, re)),
                 c='tab:orange', label=f"Rjb={X1['Rjb'][0]}, FD={X1['Focal depth'][0]}")
    axes[0].plot(X2['Mw'], np.exp(model.predict(X2, X, cluster, re)),
                 c='tab:green', label=f"Rjb={X2['Rjb'][0]}, FD={X2['Focal depth'][0]}")
    axes[0].set_xlabel('Mw')
    axes[0].set_ylabel(f'{y}')
    axes[0].legend(frameon=False)
    axes[0].set_yscale('log')

    # Plot for Rjb
    axes[1].scatter(df['Rjb'], df[y], c='tab:blue', alpha=0.5)
    axes[1].plot(X3['Rjb'], np.exp(model.predict(X3, X, cluster, re)),
                 c='tab:orange', label=f"Mw={X3['Mw'][0]}, FD={X3['Focal depth'][0]}")
    axes[1].plot(X4['Rjb'], np.exp(model.predict(X4, X, cluster, re)),
                 c='tab:green', label=f"Mw={X4['Mw'][0]}, FD={X4['Focal depth'][0]}")
    axes[1].set_xlabel('Rjb (km)')
    axes[1].set_ylabel(f'{y}')
    axes[1].legend(frameon=False)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.show()
# %%
if False:
    joblib.dump((model.trained_fe_model, model.scalerX), r'C:\Users\Sajad\OneDrive - Universidade do Minho\Record-Database\ESM-database-Europe-SJ\Tabriz-picklable-ANN-GMMs\mlp_PGA_scx.pkl')
if False:
    model = joblib.load(r'C:\Users\Sajad\OneDrive - Universidade do Minho\Record-Database\ESM-database-Europe-SJ\Tabriz-picklable-ANN-GMMs\mlp_PGA.pkl')

