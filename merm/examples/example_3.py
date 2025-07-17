# %%
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib
from pathlib import Path
# %%
base_path = Path(r"C:\Users\Sajad\Work Folder\merm_example")

X_train_processed = np.load(base_path / 'preprocess' / 'X_train_processed.npy')
y_train_log = np.load(base_path / 'preprocess' / 'y_train_log.npy')

def objective(trial):
    """
    Objective function for Optuna to optimize the MLPRegressor hyperparameters.
    This function defines the hyperparameters to be tuned and evaluates the model using cross-validation.
    """
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_l{i}', 32, 500))
    
    model_params = {
        'hidden_layer_sizes': tuple(layers),
        'activation': trial.suggest_categorical('activation', ['relu', 'logistic', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd'])}

    if model_params['solver'] == 'adam':
        model_params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    else: # sgd
        model_params['learning_rate'] = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        model_params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)

    model = MLPRegressor(random_state=42, max_iter=4000, early_stopping=True, **model_params)
    score = cross_val_score(model, X_train_processed, y_train_log, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    
    return -score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)
# %%
(base_path / 'tuned_model').mkdir(exist_ok=True)

best_params = study.best_params.copy()
n_layers = best_params.pop('n_layers')
layers = []
for i in range(n_layers):
    layer_key = f'n_units_l{i}'
    if layer_key in best_params:
        layers.append(best_params.pop(layer_key))
best_params['hidden_layer_sizes'] = tuple(layers)

tuned_model = MLPRegressor(random_state=42, max_iter=4000, early_stopping=True, **best_params)
joblib.dump(tuned_model, base_path / 'tuned_model' / 'tuned_mlp_model.joblib')
joblib.dump(study, base_path / 'tuned_model' / 'optuna_study.joblib')
print("\nStudy and tuned model saved successfully! 🎉")