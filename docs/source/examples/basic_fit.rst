.. _example_basic_fit:

Quick Start: Basic Example
======================================

For prepared data in Numpy Array, fit a basic Mixed-Effects Model using MMER
------------------------------------------------------------------------------

.. code-block:: python

   from pathlib import Path
   from sklearn.linear_model import LinearRegression
   from mmer import MixedEffectRegressor

   # %% Load Prepared Data
   base = Path(__file__).parent
   X_train = np.load(base / 'X_train.npy')
   y_train = np.load(base / 'y_train.npy')
   group_train = pd.read_csv(base / 'group_train.csv').to_numpy()

   # %% Define Fixed-Effects Model, here we use a simple Linear Regression as the fixed-effects model
   # Note: MMER can work with any scikit-learn or custom regression multioutput model that has fit and predict methods
   fe_model = LinearRegression()

   # %% Fit Mixed-Effects Model using MMER and Display Results
   model = MixedEffectRegressor(fe_model)
   result = model.fit(X_train, y_train, group_train)

   result.summary()  # Summary of the fitted model

   result.residual_correlation  # Residual correlation matrix
   result.random_effects_correlations  # Correlation matrix of random effects
   result.get_marginal_correlation()  # Marginal correlation matrix of the model
   # Note: Access the covariance matrices similarly via name like result.residual_covariance