# %%
# %reload_ext autoreload
# %autoreload 2
from meml import MEML, utils
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PowerTransformer, FunctionTransformer, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.inspection import permutation_importance

df = pd.read_csv(r'c:\Users\Sajad\Work Folder\GMM-rot\ESM-NGA-Datasets-GMM\dfx_esm.csv', dtype={'sof': str})
x_vars = ['mw', 'rrup', 'vs30', 'fd', 'sof']
cluster = 'event_id';  # cluster = 'Earthquake Name'
re = 'RE'; df['RE'] = 1.0  # add covariates for a random intercep
y_var = 'et'
df = utils.clean_outliers(df, [*x_vars[:-1], y_var], method='zscore')
data = df[[*x_vars, cluster, re, y_var]]
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
scalerx = ColumnTransformer(transformers=[
    ('num', RobustScaler(), ['mw', 'rrup', 'vs30', 'fd']),
    ('cat', OneHotEncoder(), ['sof'])])
scalery = PowerTransformer()
# scalery = RobustScaler()
# scalery = FunctionTransformer(func=np.log, inverse_func=np.exp)
# scalery = FunctionTransformer(func=np.exp, inverse_func=np.log)
# scalery = FunctionTransformer(func=np.asarray, inverse_func=np.asarray)
X = scalerx.fit_transform(train_data)
y = scalery.fit_transform(train_data[[y_var]].values)
X_val = scalerx.transform(test_data)
y_val = scalery.transform(test_data[[y_var]].values)
# %%
# fixed_model = MLPRegressor((5,), "logistic", random_state=None, solver='lbfgs', max_iter=5000)
fixed_model = LinearRegression()
# fixed_model = RandomForestRegressor(n_estimators=200, random_state=42)
model = MEML(fixed_model, max_iter=100, gll_stop=0.001, tuning=False)
model2 = MEML(fixed_model, max_iter=100, gll_stop=0.001, tuning=False)
model.fit_lme(X=X, cluster=train_data[cluster].values, Z=train_data[re].values, y=y,
          X_val=X_val, cluster_val=test_data[cluster].values, Z_val=test_data[re].values, y_val=y_val)
model2.fit(X=X, cluster=train_data[cluster].values, Z=train_data[re].values, y=y,
          X_val=X_val, cluster_val=test_data[cluster].values, Z_val=test_data[re].values, y_val=y_val)
model.summary()
model2.summary()
#%%
if False:
    X_train_summary = shap.kmeans(X, 10)
    explainer = shap.KernelExplainer(model.fe_model.predict, X_train_summary)
    shap_values = explainer.shap_values(X_val)
    # shap.summary_plot(shap_values, X_val)
    shap.summary_plot(shap_values, X_val, feature_names=scalerx.transformers_[0][2] + list(scalerx.transformers_[1][1].get_feature_names_out()))

    result = permutation_importance(model.fe_model, X_val, test_data[y_var].values, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    # Plot permutation importances
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_names = [*scalerx.transformers_[0][2], *scalerx.transformers_[1][1].get_feature_names_out()]
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=feature_names)
    ax.set_title("Permutation Importances (test set)")
    plt.show()
#%%
plot_data = train_data.copy()
plot_data['res_between'] = model.resid_re.reshape(-1,1)
plot_data['res_within'] = model.resid_unexp
if True:
    utils.qq_plot(model.resid_unexp)
    var_cols = x_vars
    res_cols = ['res_between', 'res_within', 'res_within']
    xlabel = [r'$\mathregular{M_w}$', r'$\mathregular{R_{rup}}$ (km)', r'$\mathregular{V_{s30}\,(\frac{m}{s})}$']
    ylabel = [r'$\mathregular{\eta}$', r'$\mathregular{\epsilon}$', r'$\mathregular{\epsilon}$']
    for i in range(3):
        utils.plot_joint_grid(plot_data, var_cols[i], res_cols[i], xlabel[i], ylabel[i])

    train_pred = model.predict(X, train_data[cluster].values, train_data[re].values)
    test_pred = model.predict(X_val, test_data[cluster].values, test_data[re].values)
    utils.model_performance(y.reshape(-1), train_pred, y_val.reshape(-1), test_pred, y_var)
# %%
if True:
    X1 = pd.DataFrame(columns=train_data.columns)
    X2 = pd.DataFrame(columns=train_data.columns)
    X3 = pd.DataFrame(columns=train_data.columns)
    X4 = pd.DataFrame(columns=train_data.columns)
    X5 = pd.DataFrame(columns=train_data.columns)
    X6 = pd.DataFrame(columns=train_data.columns)
    X1['mw'] = np.arange(4.7, 7.7, 0.1); X1['rrup'] = 15.0; X1['vs30'] = 400.0; X1['fd'] = 12.0; X1['sof'] = 'SS'; X1[cluster] = 'Unknown';
    X2['mw'] = np.arange(4.7, 7.7, 0.1); X2['rrup'] = 35.0; X2['vs30'] = 400.0; X2['fd'] = 12.0; X2['sof'] = 'SS'; X2[cluster] = 'Unknown';
    X3['rrup'] = np.arange(2.0, 100.0, 0.5); X3['mw'] = 7.2 ; X3['vs30'] = 400.0; X3['fd'] = 12.0; X3['sof'] = 'SS'; X3[cluster] = 'Unknown';
    X4['rrup'] = np.arange(2.0, 100.0, 0.5); X4['mw'] = 6.0; X4['vs30'] = 400.0; X4['fd'] = 12.0; X4['sof'] = 'SS'; X4[cluster] = 'Unknown';
    X5['vs30'] = np.arange(250., 1000., 10); X5['rrup'] = 15.0; X5['mw'] = 6.7; X5['fd'] = 12.0; X5['sof'] = 'SS'; X5[cluster] = 'Unknown';
    X6['vs30'] = np.arange(250., 1000., 10); X6['rrup'] = 15.0; X6['mw'] = 6.7; X6['fd'] = 12.0; X6['sof'] = 'SS'; X6[cluster] = 'Unknown';

    Xt1 = scalerx.transform(X1)
    Xt2 = scalerx.transform(X2)
    Xt3 = scalerx.transform(X3)
    Xt4 = scalerx.transform(X4)
    Xt5 = scalerx.transform(X5)
    Xt6 = scalerx.transform(X6)
    cm = utils.config_plot(lw=0.5)
    fig, axes = plt.subplots(1, 3, figsize=(18*cm, 8*cm), layout='constrained', sharey=True)
    # Plot for mw
    axes[0].scatter(df['mw'], df[y_var], c='tab:blue')
    axes[0].plot(X1['mw'].values, scalery.inverse_transform(model.predict(Xt1, X1[cluster].values, X1[re].values).reshape(-1,1)),
                 c='tab:orange', label=f"rrup={X1['rrup'][0]}, vs30={X1['vs30'][0]}")
    axes[0].plot(X2['mw'].values, scalery.inverse_transform(model.predict(Xt2, X2[cluster].values, X2[re].values).reshape(-1,1)),
                 c='tab:green', label=f"rrup={X2['rrup'][0]}, vs30={X2['vs30'][0]}")
    axes[0].set_xlabel('mw')
    axes[0].set_ylabel(f'{y_var}')
    # Plot for Rjb
    axes[1].scatter(df['rrup'], df[y_var], c='tab:blue')
    axes[1].plot(X3['rrup'].values, scalery.inverse_transform(model.predict(Xt3, X3[cluster].values, X3[re].values).reshape(-1,1)),
                 c='tab:orange', label=f"mw={X3['mw'][0]}, vs30={X3['vs30'][0]}")
    axes[1].plot(X4['rrup'].values, scalery.inverse_transform(model.predict(Xt4, X4[cluster].values, X4[re].values).reshape(-1,1)),
                 c='tab:green', label=f"mw={X4['mw'][0]}, vs30={X4['vs30'][0]}")
    axes[1].set_xlabel('rrup (km)')
    # Plot for Vs30
    axes[2].scatter(df['vs30'], df[y_var], c='tab:blue')
    axes[2].plot(X5['vs30'].values, scalery.inverse_transform(model.predict(Xt5, X5[cluster].values, X5[re].values).reshape(-1,1)),
                 c='tab:orange', label=f"mw={X5['mw'][0]}, rrup={X5['rrup'][0]}")
    axes[2].plot(X6['vs30'].values, scalery.inverse_transform(model.predict(Xt6, X6[cluster].values, X6[re].values).reshape(-1,1)),
                 c='tab:green', label=f"mw={X6['mw'][0]}, rrup={X6['rrup'][0]}")
    axes[2].set_xlabel('vs30 (m/s)')
    for ax in axes:
        ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.0, 0.95), ncol=1)
        ax.set_yscale('log')
    plt.show()
if False:
    joblib.dump(scalerx, fr'C:\Users\Future\OneDrive - Universidade do Minho (1)\PhD-Disseminations\ECCOMAS-2024\Codes-files\MEML-MLP-models\scalerx_{y_var}.joblib')
    joblib.dump({'scalery': scalery, 'model': model}, fr'C:\Users\Future\OneDrive - Universidade do Minho (1)\PhD-Disseminations\ECCOMAS-2024\Codes-files\MEML-MLP-models\meml_mlp_{y_var}.joblib')

    joblib.dump(scalerx, fr'C:\Users\Future\OneDrive - Universidade do Minho (1)\PhD-Disseminations\ECCOMAS-2024\Codes-files\MLP-models\scalerx_{y_var}.joblib')
    joblib.dump({'scalery': scalery, 'model': model.fe_model, 'var_re': model.var_re, 'var_unexp': model.var_unexp}, fr'C:\Users\Future\OneDrive - Universidade do Minho (1)\PhD-Disseminations\ECCOMAS-2024\Codes-files\MLP-models\mlp_{y_var}.joblib')

