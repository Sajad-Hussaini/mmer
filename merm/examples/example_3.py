# %%
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
import numpy as np
import joblib
from pathlib import Path
# %%
base_path = Path("/home/Sajad/WorkFolder/merm_example")

X_train = np.load(base_path / 'preprocess' / 'X_train.npy')
y_train = np.load(base_path / 'preprocess' / 'y_train.npy')
group_train = np.load(base_path / 'preprocess' / 'group_train.npy', allow_pickle=True)[:, 0]

def objective(trial):
    """
    Objective function for Optuna to optimize the MLPRegressor hyperparameters.
    This function defines the hyperparameters to be tuned and evaluates the model using cross-validation.
    """
    n_layers = trial.suggest_int('n_layers', 1, 2)
    layers = []

    units = trial.suggest_int('n_units_l0', 5, 50)
    layers.append(units)
    for i in range(1, n_layers):
        units = trial.suggest_int(f'n_units_l{i}', max(5, int(units * 0.2)), min(50, int(units * 1.8)))
        layers.append(units)
    
    model_params = {
        'hidden_layer_sizes': tuple(layers),
        'activation': trial.suggest_categorical('activation', ['relu', 'logistic', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-2, 10.0, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        }

    fullmodel = MLPRegressor(random_state=42, max_iter=2000, early_stopping=True, solver='adam', **model_params)
    kf = GroupKFold(n_splits=10, shuffle=True, random_state=42)
    try:
        score = cross_val_score(fullmodel, X_train, y_train, groups=group_train, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1).mean()
        if not np.isfinite(score):
            return float('inf')
                 
    except Exception:
        return float('inf')
    
    return -score

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=10))

study.optimize(objective, n_trials=300)
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

tuned_model = MLPRegressor(random_state=42, max_iter=2000, early_stopping=True, solver='adam', **best_params)
joblib.dump(tuned_model, base_path / 'tuned_model' / 'tuned_mlp_model.joblib')
joblib.dump(study, base_path / 'tuned_model' / 'optuna_study.joblib')
print("\nStudy and tuned model saved successfully! 🎉")