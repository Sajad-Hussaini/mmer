.. _example_basic_fit:

Quick Start: Basic Template for Mixed-Effects Modeling
=======================================================

This template guides you through fitting a Mixed-Effects Model using MMER. Each section explains the requirements and options for flexible, robust modeling.

1. Prepare Your Data (Numpy Arrays Required)
---------------------------------------------
You must have your data preprocessed as numpy arrays:

- `X_train`: Covariates/features, shape (n_samples, n_features)
- `y_train`: Outcomes/targets, shape (n_samples, n_outputs)
- `group_train`: Grouping variable, shape (n_samples,) or (n_samples, n_grouping_vars)

.. code-block:: python

   import numpy as np
   import pandas as pd
   from pathlib import Path

   base = Path(__file__).parent
   X_train = np.load(base / 'X_train.npy')
   y_train = np.load(base / 'y_train.npy')
   group_train = pd.read_csv(base / 'group_train.csv').to_numpy()

2. Choose a Fixed-Effects Model
--------------------------------
You can use any multi-output regressor with `fit` and `predict` methods:

- Simple parametric: `sklearn.linear_model.LinearRegression()`
- Custom parametric: Your own class with `fit`/`predict`
- Nonparametric/ML: Any model (e.g., PyTorch, PINN, RandomForestRegressor, etc.)

.. code-block:: python

   from sklearn.linear_model import LinearRegression
   # Or use your own model class
   fe_model = LinearRegression()

3. Fit the Mixed-Effects Model
-------------------------------
Pass your fixed-effects model and data to `MixedEffectRegressor`. Default values are safe for most use cases.

.. code-block:: python

   from mmer import MixedEffectRegressor
   model = MixedEffectRegressor(fe_model)
   result = model.fit(X_train, y_train, group_train)

4. Summarize and Interpret Results
----------------------------------
The result object provides:

- `result.summary()`: Print a summary table of the fitted model
- `result.residual_correlation`: Residual correlation matrix
- `result.random_effects_correlations`: Correlation matrix of random effects
- `result.get_marginal_correlation()`: Marginal correlation matrix
- Covariance matrices: e.g., `result.residual_covariance`

.. code-block:: python

   print(result.summary())
   print(result.residual_correlation)
   print(result.random_effects_correlations)
   print(result.get_marginal_correlation())
   # Covariance: result.residual_covariance