# %%
import numpy as np
import joblib
from pathlib import Path
from merm import MERM
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# %%
base_path = Path("/home/Sajad/WorkFolder/merm_example")

X_train = np.load(base_path / 'preprocess' / 'X_train.npy')
y_train = np.load(base_path / 'preprocess' / 'y_train.npy')
group_train = np.load(base_path / 'preprocess' / 'group_train.npy', allow_pickle=True)
fe_model = joblib.load(base_path / 'tuned_model' / 'tuned_mlp_model.joblib')
# %%
model = MERM(fe_model, 60, 1e-4, 50, 50, True, 'bste', 'norm', 16, 'loky')
result = model.fit(X_train, y_train, group_train, None)
result.summary()
# %%
(base_path / 'fitted_model').mkdir(exist_ok=True)
joblib.dump(result, base_path / 'fitted_model' / 'fitted_model_mlp.joblib')
print("Model saved successfully.")