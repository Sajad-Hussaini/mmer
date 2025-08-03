# %%
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# %%
base_path = Path(r"C:\Users\Sajad\Work Folder\merm_example")

X_train = np.load(base_path / 'preprocess' / 'X_train.npy')
X_test = np.load(base_path / 'preprocess' / 'X_test.npy')
y_train = np.load(base_path / 'preprocess' / 'y_train.npy')
y_test = np.load(base_path / 'preprocess' / 'y_test.npy')
group_train = np.load(base_path / 'preprocess' / 'group_train.npy', allow_pickle=True)
group_test = np.load(base_path / 'preprocess' / 'group_test.npy', allow_pickle=True)
preprocessor_x = joblib.load(base_path / 'preprocess' / 'preprocessor_x.joblib')
preprocessor_y = joblib.load(base_path / 'preprocess' / 'preprocessor_y.joblib')
result = joblib.load(base_path / 'fitted_model' / 'fitted_model_mlp.joblib')

result.summary()
y_pred_train_prc = result.predict(X_train)
y_pred_test_prc = result.predict(X_test)
# %%
periods = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.6, 0.75, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5, 9.0, 10.0])
sa_labels = [f"SA({T}s)" for T in periods]
plt.figure(figsize=(12, 10))
sns.heatmap(result.resid.to_corr(),  annot=False, cmap='coolwarm',
            linewidths=.5, xticklabels=sa_labels, yticklabels=sa_labels)
plt.title('Between-Station Correlation Matrix of SA')
plt.show()
# %%
def evaluate_performance(y_true_log, y_pred_log):
    """Calculates and returns a DataFrame of performance metrics."""
    # --- Metrics on Log Scale ---
    r2 = r2_score(y_true_log, y_pred_log)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    
    # --- Metrics on Original Scale (more intuitive) ---
    y_true_orig = np.exp(y_true_log)
    y_pred_orig = np.exp(y_pred_log)
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    
    # --- Combine into a simple DataFrame ---
    metrics = {
        'R2 Score': r2,
        'MAE (log)': mae_log,
        'RMSE (log)': rmse_log,
        'MAE (original g)': mae_orig,
        'RMSE (original g)': rmse_orig
    }
    return pd.Series(metrics)
# %%
train_metrics = evaluate_performance(y_train, y_pred_train_prc)
test_metrics = evaluate_performance(y_test, y_pred_test_prc)

# Display results side-by-side for easy comparison
comparison_df = pd.DataFrame({'Train': train_metrics, 'Test': test_metrics})
print("--- Performance Metrics Comparison ---")
display(comparison_df)
# %%
outputs_to_plot = [0, 5, 10, 15] 
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, ax in zip(outputs_to_plot, axes):
    actual = y_test[:, i]
    predicted = y_pred_test_prc[:, i]
    
    sns.scatterplot(x=actual, y=predicted, alpha=0.5, ax=ax)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    ax.set_title(f'Test Set: Output Variable {i+1}')
    ax.set_xlabel('Actual Values (log)')
    ax.set_ylabel('Predicted Values (log)')
    ax.grid(True)

plt.tight_layout()
plt.show()
# %%
spectral_periods = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.6, 0.75, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5, 9.0, 10.0])

# Choose a few records from the test set to plot
records_to_plot = [10, 50, 100, 150]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Use the original scale for physical interpretation
y_test_original = np.exp(y_test)
y_pred_original = np.exp(y_pred_test_prc)

for i, ax in zip(records_to_plot, axes):
    # Plot observed spectrum
    ax.plot(spectral_periods, y_test_original[i:i+10, :].T, 'o-', label='Observed SA')
    # Plot predicted spectrum
    ax.plot(spectral_periods, y_pred_original[i:i+10, :].T, 'r--', label='Predicted SA')

    # Use log-log scale, which is standard for response spectra
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title(f'Response Spectrum for Test Record #{i}')
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Spectral Acceleration (g)')
    # ax.legend()
    ax.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()
# %%# Choose one response to analyze, e.g., T=1.0s
# Let's assume the 10th column is SA at 1.0s
period_index = 10 
period_label = 'SA (T=1.0s)'

# Use data on the original scale for intuition
y_test_original = (y_test)
y_pred_original = (y_pred_test_prc)

# --- Plot against Magnitude (Mw) ---
plt.figure(figsize=(10, 6))
# Plot the real data
sns.scatterplot(x=X_test[:, 0], y=y_test_original[:, period_index], alpha=0.3, label='Observed Data')
# Plot the model's predictions
sns.scatterplot(x=X_test[:, 0], y=y_pred_original[:, period_index], alpha=0.3, color='red', label='Predicted Value')

plt.title(f'{period_label} vs. Magnitude')
plt.xlabel('Magnitude (Mw)')
plt.ylabel(period_label)
plt.legend()
plt.grid(True)
plt.show()