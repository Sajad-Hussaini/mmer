.. _example_basic_fit:

Quick Start: Basic Example
======================================

Prepare Data
-----------------------------------------------------

.. code-block:: python

   from pathlib import Path
   from sklearn.linear_model import LinearRegression
   from mmer import MixedEffectRegressor

   base = Path(__file__).parent
   X_train = np.load(base / 'X_train.npy')
   y_train = np.load(base / 'y_train.npy')
   group_train = pd.read_csv(base / 'group_train.csv').to_numpy()

   model = MixedEffectRegressor(LinearRegression(), 50, 1e-3, 1, 50, 50, True, 'bste', -1, 'loky')
   result = model.fit(X_train, y_train, group_train, None)

   result.summary()  # Summary of the fitted model