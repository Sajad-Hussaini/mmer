from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor

base_path = Path(r"C:\Users\Sajad\Work Folder\merm_example")
# %% Loading the processed NumPy arrays and the fitted preprocessor
X_train_processed = np.load(base_path / 'preprocess' / 'X_train_processed.npy')
X_test_processed = np.load(base_path / 'preprocess' / 'X_test_processed.npy')
y_train_log = np.load(base_path / 'preprocess' / 'y_train_log.npy')
y_test_log = np.load(base_path / 'preprocess' / 'y_test_log.npy')
preprocessor = joblib.load(base_path / 'preprocess' / 'preprocessor.joblib')
fe_model = MLPRegressor(random_state=42, max_iter=1000)
# %% Broader search for hyperparameters using RandomizedSearchCV
param_dist = {
    'hidden_layer_sizes': [(5,), (15,), (25,), (50,), (100,), (5, 5), (15, 15), (25, 25), (50, 50), (100, 50), (100, 100), (5, 5, 5), (15, 15, 15)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': np.logspace(-4, 0, 5),
    'learning_rate': ['constant', 'adaptive']}
random_tuner = RandomizedSearchCV(fe_model, param_dist, n_iter=200, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42).fit(X_train_processed, y_train_log)
print(f"Best parameters found: {random_tuner.best_params_}")
# %% Focused search for hyperparameters using GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(5,), (10,), (20,), (50,), (100,), (5, 5), (10, 10), (50, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01, 0.1]}
tuner = GridSearchCV(fe_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).fit(X_train_processed, y_train_log)
# %% Save the best model
(base_path / 'best_model').mkdir(exist_ok=True)
joblib.dump(tuner.best_estimator_, base_path / 'best_model' / 'best_mlp_model.joblib')