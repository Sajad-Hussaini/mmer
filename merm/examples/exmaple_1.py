# This module preprocess data
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
categorical_vars = ['SoF']
numeric_vars = ['EarthquakeMagnitude', 'ClstD(km)', 'Vs30(m/s)', 'HypocenterDepth(km)']
x_vars = numeric_vars + categorical_vars
y_vars = [f"sax{v:.2f}" for v in np.arange(0.1, 10.1, 0.1)]
group_vars = ['EarthquakeName', 'StationName']
all_needed_cols = x_vars + y_vars + group_vars
dtype_mapping = {'EarthquakeName': 'str'}
df = pd.read_csv(base_path / "df_nga_sa.csv", usecols=all_needed_cols, dtype=dtype_mapping)
df.dropna(subset=y_vars, inplace=True)
print(f"Shape with no missing y values: {df.shape}")
# %% Grouped-Splitting the dataset
# For multi-grouped-splitting combine the groups df["group"] = df["EarthquakeName"] + "_" + df["StationName"]
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