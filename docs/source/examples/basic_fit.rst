.. _example_basic_fit:

Quick Start: Basic Example
======================================

For prepared data in Numpy Array, fit a basic Mixed-Effects Model using MMER
------------------------------------------------------------------------------

.. code-block:: python

   import numpy as np
   from pathlib import Path
   from sklearn.linear_model import LinearRegression
   from mmer import MixedEffectRegressor

   # %% Load Prepared Data
   base = Path(__file__).parent
   X_train = np.load(base / 'X_train.npy')
   y_train = np.load(base / 'y_train.npy')
   group_train = np.loadtxt(base / 'group_train.csv', delimiter=',', dtype=int)
   if group_train.ndim == 1:
      group_train = group_train[:, None]

   # %% Define Fixed-Effects Model
   fe_model = LinearRegression()

   # %% Fit Mixed-Effects Model using MMER and Display Results
   model = MixedEffectRegressor(fe_model, 50, 1e-3, 1, 50, 50, True, 'bste', -1, 'loky')
   result = model.fit(X_train, y_train, group_train, None)

   result.summary()  # Summary of the fitted model