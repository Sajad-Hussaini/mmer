.. _example_statsmodel_comparison:

Quick Start: Basic Numerical Comparison with Statsmodels
=========================================================

Comparing Linear Mixed-Effects Models using MMER and Statsmodels For Numerical Validation
-------------------------------------------------------------------------------------------

.. code-block:: python

   import time
   import numpy as np
   import pandas as pd
   import statsmodels.api as sm
   from statsmodels.regression.mixed_linear_model import MixedLM
   
   from mmer import MixedEffectRegressor
   from sklearn.linear_model import LinearRegression

   # =========================================================
   # 1. Prepare Simulated Data
   # =========================================================
   np.random.seed(42)
   n_samples = 100_000
   n_groups = 200
   n_features = 5

   groups = np.repeat(np.arange(n_groups), n_samples // n_groups)[:, None]
   X = np.random.randn(n_samples, n_features)
   
   # Ground truth variances
   true_resid_var = 0.3
   true_re_var = 0.2
   
   # True coefficients
   true_coef = np.random.randn(n_features)
   true_intercept = 1.5

   # Generate group effects and target
   b_k = np.random.normal(0, np.sqrt(true_re_var), n_groups)
   b_k_expanded = b_k[groups.flatten()]
   y = true_intercept + X @ true_coef + b_k_expanded + np.random.normal(0, np.sqrt(true_resid_var), n_samples)
   
   # MMER expects 2D targets natively for multi-output support
   y_mmer = y[:, None] 

   # =========================================================
   # 2. Fit using MMER
   # =========================================================
   print("--- Fitting MMER ---")
   fe_model = LinearRegression()
   
   # Using kwargs for clean, readable instantiation
   mmer_model = MixedEffectRegressor(
       fixed_effects_model=fe_model, 
       max_iter=100, 
       tol=1e-6
   )
   
   start_t = time.time()
   result_mmer = mmer_model.fit(X, y_mmer, groups)
   mmer_time = time.time() - start_t
   
   print(f"MMER Fit Time: {mmer_time:.2f} seconds")
   # Optional: built-in summary (if implemented)
   # result_mmer.summary()
   
   print("\nMMER Fixed Effects:")
   print(f"  Intercept: {float(result_mmer.fe_model.intercept_):.4f}")
   for i, coef in enumerate(np.ravel(result_mmer.fe_model.coef_)):
       print(f"  x{i:d}:       {coef:.4f}")
   
   print("\nMMER Variance Components:")
   print(f"  Residual:  {float(result_mmer.residual_covariance):.4f}")
   print(f"  Random:    {float(result_mmer.random_effects_covariances[0]):.4f}")

   # =========================================================
   # 3. Fit using Statsmodels MixedLM
   # =========================================================
   print("\n--- Fitting Statsmodels MixedLM ---")
   df = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_features)])
   df['y'] = y
   df['group'] = groups.flatten()
   
   # Add explicit intercept column for statsmodels
   exog = sm.add_constant(df[[f'x{i}' for i in range(n_features)]])
   
   # By default, MixedLM automatically includes a random intercept
   sm_model = MixedLM(df['y'], exog, groups=df['group'])
   
   start_t = time.time()
   result_sm = sm_model.fit()
   sm_time = time.time() - start_t
   
   print(f"Statsmodels Fit Time: {sm_time:.2f} seconds\n")
   print(result_sm.summary())
