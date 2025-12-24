.. _example_statsmodel_comparison:

Quick Start: Basic Numerical Comparison with Statsmodels
=========================================================

Comparing Linear Mixed-Effects Models using MMER and Statsmodels For Numerical Validation
-------------------------------------------------------------------------------------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from mmer import MixedEffectRegressor
   from sklearn.linear_model import LinearRegression
   import statsmodels.api as sm
   from statsmodels.regression.mixed_linear_model import MixedLM

   # %% Prepare Data For a Univariate Mixed-Effects LinearRegression
   np.random.seed(42)
   n, o_k = 100000, 200
   groups = np.repeat(np.arange(o_k), n // o_k)[:, None]
   X = np.random.randn(n, 5)
   true_phi = 0.3  # Residual variance
   true_tau = 0.2  # Random effects variance
   b_k = np.random.normal(0, np.sqrt(true_tau), o_k)
   Z_k = np.zeros((n, o_k))
   for i, g in enumerate(groups): Z_k[i, g] = 1
   cof = np.random.randn(5)
   fX = X @ cof  # Fixed effects
   y = fX + Z_k @ b_k + np.random.normal(0, np.sqrt(true_phi), n)
   y = y[:, None]

   # %% Perform MMER on the Data and Display Results
   fe_model = LinearRegression()
   model = MixedEffectRegressor(fe_model, 500, 1e-8, 1, 50, 50, True, 'bste', 1, 'loky')
   result = model.fit(X, y, groups, None)
   result.summary()
   print("\nFixed Effects Coefficients:")
   print("-" * 50)
   print(f"Intercept: {result.fe_model.intercept_:.6f}")
   print("\nCoefficients:")
   feature_names = ["Mw", "Ztor", "ln_Rrup", "ln_VS30", "Fm0", "Fm1", "Fm2", "Fm3"]
   for name, coef in zip(feature_names, result.fe_model.coef_):
      print(f"  {name:12s}: {coef:10.6f}")
   
   # %% Perform Statsmodels MixedLM on the Data and Display Results
   df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
   df['y'] = y
   df['group'] = groups.flatten()
   exog = sm.add_constant(df[[f'x{i}' for i in range(5)]])  # add intercept
   exog_re = np.ones((n, 1))  # Random intercept
   model2 = MixedLM(df['y'], exog, groups=df['group'], exog_re=exog_re)
   result2 = model2.fit()
   print(result2.summary())
