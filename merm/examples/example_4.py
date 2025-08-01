# %%
import numpy as np
import joblib
from pathlib import Path
from merm import MERM
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# %%
base_path = Path(r"C:\Users\Sajad\Work Folder\merm_example")

X_train_mlp = np.load(base_path / 'preprocess' / 'X_train_mlp.npy')
y_train_log = np.load(base_path / 'preprocess' / 'y_train_log.npy')
group_train = np.load(base_path / 'preprocess' / 'group_train.npy', allow_pickle=True)
fe_model = joblib.load(base_path / 'tuned_model' / 'tuned_mlp_model.joblib')
# %%
model = MERM(fe_model, 60, 1e-5, 50, 50, False, 'bste', 'parameters', 16, 'loky')
result = model.fit(X_train_mlp, y_train_log, group_train, None)
result.summary()
# %%
(base_path / 'fitted_model').mkdir(exist_ok=True)
joblib.dump(result, base_path / 'fitted_model' / 'fitted_model_mlpregressor.joblib')
print("Model saved successfully.")