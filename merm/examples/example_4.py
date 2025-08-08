# %%
import numpy as np
import joblib
from pathlib import Path
from merm import MERM
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
# from dask.distributed import Client
# client = Client('tcp://192.168.219.38:8786')
# %%
base_path = Path("/home/Sajad/WorkFolder/merm_example")
X_train = np.load(base_path / 'preprocess' / 'X_train.npy')
y_train = np.load(base_path / 'preprocess' / 'y_train.npy')
group_train = np.load(base_path / 'preprocess' / 'group_train.npy', allow_pickle=True)
fe_model = joblib.load(base_path / 'tuned_model' / 'tuned_mlp_model.joblib')
# %%
model = MERM(fe_model, 60, 1e-3, 50, 50, True, 'bste', 'log_lh', -1, 'loky')
result = model.fit(X_train, y_train, group_train, None)
result.summary()
# %%
(base_path / 'fitted_model').mkdir(exist_ok=True)
joblib.dump(result, base_path / 'fitted_model' / 'fitted_model_mlp.joblib')
print("Model saved successfully.")

# client.close()