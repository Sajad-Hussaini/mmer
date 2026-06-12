.. _example_basic_fit:

Quick Start: Basic Template for Mixed-Effects Modeling
=======================================================

This template guides you through fitting a Mixed-Effects Model using MMER. Each section explains the requirements and options for flexible, robust modeling.

1. Prepare Your Data (Numpy Arrays Required)
---------------------------------------------
You must have your data preprocessed as numpy arrays:

- ``X_train``: Covariates/features, shape ``(n_samples, n_features)``
- ``y_train``: Outcomes/targets, shape ``(n_samples, n_outputs)``
- ``group_train``: Grouping factors, shape ``(n_samples, n_groups)`` — must be 2-dimensional

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
You can use any multi-output regressor with ``fit`` and ``predict`` methods:

- Simple parametric: ``LinearRegression``
- Custom parametric: Your own class with ``fit``/``predict``
- Nonparametric/ML: Any model (e.g., a neural network, gradient boosted trees, etc.)

.. code-block:: python

   from sklearn.linear_model import LinearRegression
   # Or use your own model class
   fe_model = LinearRegression()

3. Fit the Mixed-Effects Model
-------------------------------
Pass your fixed-effects model and data to ``MixedEffectEstimator``. Default values are safe for most use cases.

.. code-block:: python

   from mmer import MixedEffectEstimator
   model = MixedEffectEstimator(fe_model)
   result = model.fit(X_train, y_train, group_train)

4. Summarize and Interpret Results
----------------------------------
The result object provides:

- `result.summary()`: Returns a summary string of the fitted model
- `result.R_corr`: Residual correlation matrix
- `result.G_corr`: Correlation matrices of random effects
- `result.R.matrix`: Residual covariance matrix
- `result.G[k].matrix`: Random effects covariance matrix for group `k`

.. code-block:: python

   print(result.summary())
   print("Residual Correlation:", result.R_corr)
   print("Random Effects Correlation:", result.G_corr)
   print("Residual Covariance:", result.R.matrix)