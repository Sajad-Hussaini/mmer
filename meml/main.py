# %%
%reload_ext autoreload
%autoreload 2
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
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r'c:\Users\Sajad\Work Folder\GMM-rot\ESM-NGA-Datasets-GMM\dfx_nga.csv', dtype={'sof': str})
x_vars = ['mw', 'rrup', 'vs30', 'fd', 'sof']
group_var = 'event_id'
y_var = 'et'

# Data preparation and cleaning
# df = utils.clean_outliers(df, [*x_vars[:-1], y_var], method='zscore')
df = df.dropna(subset=x_vars)
data = df[[*x_vars, group_var, y_var]]
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Define preprocessing pipelines
scalerx = ColumnTransformer(transformers=[
    ('num', RobustScaler(), [*x_vars[:-1]]),
    ('cat', OneHotEncoder(), [x_vars[-1]])], remainder='drop')
scalery = PowerTransformer()
# scalery = FunctionTransformer(func=np.log, inverse_func=np.exp)
x = scalerx.fit_transform(train_data)
y = scalery.fit_transform(train_data[[y_var]]).ravel()
groups = train_data[group_var].values

x_val = scalerx.transform(test_data)
y_val = scalery.transform(test_data[[y_var]]).ravel()
groups_val = test_data[group_var].values
# %%
# fe_model = MLPRegressor((5,), "logistic", random_state=None, solver='lbfgs', max_iter=5000)
base_model = MLPRegressor(random_state=42)
param_grid = {
    'hidden_layer_sizes': [(5,), (10,), (15,), (5, 5)],
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'solver': ['adam', 'lbfgs']}
print("Starting hyperparameter tuning...")
start_tune = time.perf_counter()
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(x, y)
fe_model = grid_search.best_estimator_
print(f"Tuning time: {time.perf_counter() - start_tune:.2f} s")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_:.4f}")
# %%
# fe_model = MLPRegressor((5,), "logistic", alpha=1.0, random_state=42, solver='lbfgs', max_iter=1000)
fe_model = LinearRegression()
start = time.perf_counter()
model = MEML(fe_model, max_iter=120, gll_limit=0.000001)
model.fit(x, groups, y, x_val, groups_val, y_val, method='mixedlm')
print(f"Training time: {time.perf_counter() - start:.2f} s")
model.summary()
#%%
plot_data = train_data.copy()
plot_data['res_between'] = model.resid_re.ravel()
plot_data['res_within'] = model.resid_unexp
if True:
    utils.qq_plot(model.resid_unexp)
    res_cols = ['res_between', 'res_within', 'res_within']
    xlabel = [r'$\mathregular{M_w}$', r'$\mathregular{R_{rup}}$ (km)', r'$\mathregular{V_{s30}\,(\frac{m}{s})}$']
    ylabel = [r'$\mathregular{\eta}$', r'$\mathregular{\epsilon}$', r'$\mathregular{\epsilon}$']
    for i in range(3):
        utils.plot_joint_grid(plot_data, x_vars[i], res_cols[i], xlabel[i], ylabel[i])
    # with added RE
    train_pred = model.predict(x, groups)
    test_pred = model.predict(x_val, groups_val)
    utils.model_performance(y, train_pred, y_val, test_pred, y_var)
    # only population mean
    train_pred = model.fe_model.predict(x)
    test_pred = model.fe_model.predict(x_val)
    utils.model_performance(y, train_pred, y_val, test_pred, y_var)
# %%
if True:
    x1 = pd.DataFrame(columns=train_data.columns)
    x2 = pd.DataFrame(columns=train_data.columns)
    x3 = pd.DataFrame(columns=train_data.columns)
    x4 = pd.DataFrame(columns=train_data.columns)
    x5 = pd.DataFrame(columns=train_data.columns)
    x6 = pd.DataFrame(columns=train_data.columns)
    x1['mw'] = np.arange(4.7, 7.7, 0.1); x1['rrup'] = 15.0; x1['vs30'] = 400.0; x1['fd'] = 12.0; x1['sof'] = 'strike slip'
    x2['mw'] = np.arange(4.7, 7.7, 0.1); x2['rrup'] = 35.0; x2['vs30'] = 400.0; x2['fd'] = 12.0; x2['sof'] = 'strike slip'
    x3['rrup'] = np.arange(2.0, 100.0, 0.5); x3['mw'] = 7.2 ; x3['vs30'] = 400.0; x3['fd'] = 12.0; x3['sof'] = 'strike slip'
    x4['rrup'] = np.arange(2.0, 100.0, 0.5); x4['mw'] = 6.0; x4['vs30'] = 400.0; x4['fd'] = 12.0; x4['sof'] = 'strike slip'
    x5['vs30'] = np.arange(250., 1000., 10); x5['rrup'] = 15.0; x5['mw'] = 6.7; x5['fd'] = 12.0; x5['sof'] = 'strike slip'
    x6['vs30'] = np.arange(250., 1000., 10); x6['rrup'] = 15.0; x6['mw'] = 6.7; x6['fd'] = 12.0; x6['sof'] = 'strike slip'

    xt1 = scalerx.transform(x1)
    xt2 = scalerx.transform(x2)
    xt3 = scalerx.transform(x3)
    xt4 = scalerx.transform(x4)
    xt5 = scalerx.transform(x5)
    xt6 = scalerx.transform(x6)

    cm = utils.config_plot(lw=0.5)
    fig, axes = plt.subplots(1, 3, figsize=(18*cm, 8*cm), layout='constrained', sharey=True)
    # Plot for mw
    axes[0].scatter(df['mw'], df[y_var], c='tab:blue')
    axes[0].plot(x1['mw'].values, scalery.inverse_transform(model.fe_model.predict(xt1)[..., None]),
                 c='tab:orange', label=f"rrup={x1['rrup'][0]}, vs30={x1['vs30'][0]}")
    axes[0].plot(x2['mw'].values, scalery.inverse_transform(model.fe_model.predict(xt2)[..., None]),
                 c='tab:green', label=f"rrup={x2['rrup'][0]}, vs30={x2['vs30'][0]}")
    axes[0].set_xlabel('mw')
    axes[0].set_ylabel(f'{y_var}')
    # Plot for Rjb
    axes[1].scatter(df['rrup'], df[y_var], c='tab:blue')
    axes[1].plot(x3['rrup'].values, scalery.inverse_transform(model.fe_model.predict(xt3)[..., None]),
                 c='tab:orange', label=f"mw={x3['mw'][0]}, vs30={x3['vs30'][0]}")
    axes[1].plot(x4['rrup'].values, scalery.inverse_transform(model.fe_model.predict(xt4)[..., None]),
                 c='tab:green', label=f"mw={x4['mw'][0]}, vs30={x4['vs30'][0]}")
    axes[1].set_xlabel('rrup (km)')
    # Plot for Vs30
    axes[2].scatter(df['vs30'], df[y_var], c='tab:blue')
    axes[2].plot(x5['vs30'].values, scalery.inverse_transform(model.fe_model.predict(xt5)[..., None]),
                 c='tab:orange', label=f"mw={x5['mw'][0]}, rrup={x5['rrup'][0]}")
    axes[2].plot(x6['vs30'].values, scalery.inverse_transform(model.fe_model.predict(xt6)[..., None]),
                 c='tab:green', label=f"mw={x6['mw'][0]}, rrup={x6['rrup'][0]}")
    axes[2].set_xlabel('vs30 (m/s)')
    for ax in axes:
        ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(0.0, 0.95), ncol=1)
        ax.set_yscale('log')
    plt.show()
if False:
    joblib.dump(scalerx, fr'\scalerx_{y_var}.joblib')
    joblib.dump({'scalery': scalery, 'model': model}, fr'\meml_mlp_{y_var}.joblib')

    joblib.dump(scalerx, fr'\scalerx_{y_var}.joblib')
    joblib.dump({'scalery': scalery, 'model': model.fe_model, 'var_re': model.var_re, 'var_unexp': model.var_unexp}, fr'\mlp_{y_var}.joblib')