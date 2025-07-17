# This module preprocess data
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

base_path = Path(r"C:\Users\Sajad\Work Folder\merm_example")

periods = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.6, 0.75, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5, 9.0, 10.0])
y_vars = [f"T{v:.3f}S" for v in periods]
categorical_vars = ['Mechanism Based on Rake Angle']
numeric_vars = ['Earthquake Magnitude', 'ClstD (km)', 'Vs30 (m/s) selected for analysis', 'Hypocenter Depth (km)']
x_vars = numeric_vars + categorical_vars
group_vars = ['Earthquake Name', 'Station Name']
all_needed_cols = x_vars + y_vars + group_vars
dtype_mapping = {'Earthquake Name': 'str', 'Station Name': 'str',
                 'Mechanism Based on Rake Angle': 'str'}

df = pd.read_csv(base_path / "nga_rotD50.csv", usecols=all_needed_cols, dtype=dtype_mapping)
df[y_vars] = df[y_vars].replace(-999.0, np.nan)
df.dropna(subset=y_vars, inplace=True)
print(f"Shape with no missing y values: {df.shape}")
# %% Check if there are multiple recordings for the same station to be considered as random effects
station_counts = df['Station Name'].value_counts()
multiple_records = station_counts[station_counts > 1]
print(f"Stations with multiple recordings: {len(multiple_records)}")
# %% Grouped-Splitting the dataset based on major group
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(df[numeric_vars], df[y_vars], groups=df[group_vars[0]]))
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]
print(f"Training set: {train_df.shape}, Test set: {test_df.shape}")
# %% Exploratory data analysis of training set
# CHECK 1: Visualize the target variable's distribution, the need for a log transformation.

# CHECK 2: Visualize correlations between numeric predictors, check for multicollinearity.
# %% preprocessing pipelines
numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_pipeline, numeric_vars), ('cat', categorical_pipeline, categorical_vars)])

X_train_processed = preprocessor.fit_transform(train_df[x_vars])
X_test_processed = preprocessor.transform(test_df[x_vars])

y_train = np.log(train_df[y_vars].to_numpy())
y_test = np.log(test_df[y_vars].to_numpy())

# %% Saving the processed NumPy arrays and the fitted preprocessor
(base_path / 'preprocess').mkdir(exist_ok=True)
np.save(base_path / 'preprocess' / 'X_train_processed.npy', X_train_processed)
np.save(base_path / 'preprocess' / 'X_test_processed.npy', X_test_processed)
np.save(base_path / 'preprocess' / 'y_train_log.npy', y_train)
np.save(base_path / 'preprocess' / 'y_test_log.npy', y_test)
np.save(base_path / 'preprocess' / 'group_train.npy', train_df[group_vars].to_numpy(), allow_pickle=True)
np.save(base_path / 'preprocess' / 'group_test.npy', test_df[group_vars].to_numpy(), allow_pickle=True)

joblib.dump(preprocessor, base_path / 'preprocess' / 'preprocessor.joblib')

print("All processed data and the preprocessor have been saved successfully! 🎉")