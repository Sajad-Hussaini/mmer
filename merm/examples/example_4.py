# %%
import numpy as np
import joblib
from pathlib import Path
from merm import MERM
# %%
base_path = Path(r"C:\Users\Sajad\Work Folder\merm_example")

X_train_processed = np.load(base_path / 'preprocess' / 'X_train_processed.npy')
X_test_processed = np.load(base_path / 'preprocess' / 'X_test_processed.npy')
y_train_log = np.load(base_path / 'preprocess' / 'y_train_log.npy')
y_test_log = np.load(base_path / 'preprocess' / 'y_test_log.npy')
group_train = np.load(base_path / 'preprocess' / 'group_train.npy', allow_pickle=True)
group_test = np.load(base_path / 'preprocess' / 'group_test.npy', allow_pickle=True)
preprocessor = joblib.load(base_path / 'preprocess' / 'preprocessor.joblib')
fe_model = joblib.load(base_path / 'tuned_model' / 'tuned_mlp_model.joblib')

model = MERM(fe_model, 40, 1e-5, 5, 5, 10, 'loky')
result = model.fit(X_train_processed, y_train_log, group_train, None)
result.summary()
# %%
(base_path / 'fitted_model').mkdir(exist_ok=True)
joblib.dump(result, base_path / 'fitted_model' / 'fitted_model.joblib')