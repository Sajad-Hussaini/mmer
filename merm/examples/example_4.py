# %%
import numpy as np
import joblib
from pathlib import Path
from merm import MERM
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import timeit
import time
# %%
base_path = Path(r"C:\Users\Sajad\Work Folder\merm_example")

X_train_mlp = np.load(base_path / 'preprocess' / 'X_train_mlp.npy')
y_train_log = np.load(base_path / 'preprocess' / 'y_train_log.npy')
group_train = np.load(base_path / 'preprocess' / 'group_train.npy', allow_pickle=True)
fe_model = joblib.load(base_path / 'tuned_model' / 'tuned_mlp_model.joblib')
a,b,c = MERM(LinearRegression(), 60, 1e-5, 25, 25, False, 'bste', 1, 'loky').prepare_data(X_train_mlp, y_train_log, group_train, None)
# %%

re = b[1]
x_vec = np.random.rand(re.m * re.q * re.o)
# %%
# runtime = timeit.timeit(lambda: re.kronZ_D_matvec(x_vec), number=1)
# runtime2 = timeit.timeit(lambda: re.kronZ_D_matvec2(x_vec), number=1)
# runtime3 = timeit.timeit(lambda: re.kronZ_D_matvec3(x_vec), number=1)
# print('Time:', runtime)
# print('Time:', runtime2)
# print('Time:', runtime3)
# %%
model = MERM(LinearRegression(), 60, 1e-5, 25, 25, False, 'bste', 1, 'loky')
result = model.fit(X_train_mlp, y_train_log, group_train, None)
result.summary()
# %%
(base_path / 'fitted_model').mkdir(exist_ok=True)
joblib.dump(result, base_path / 'fitted_model' / 'fitted_model_mlpregressor.joblib')
print("Model saved successfully.")