.. _example_ensemble_fit:

Ensemble Mixed-Effects Modeling for Epistemic Uncertainty
=========================================================

The ``EnsembleMixedModel`` is a powerful aggregation wrapper. By training multiple independent Mixed-Effects Models (e.g., using bootstrapped samples of your data or different random seeds), you can wrap them in the Ensemble object. This allows you to effortlessly compute the expected mean and epistemic standard deviation (uncertainty) across all models.

1. Prepare Your Data
--------------------
.. code-block:: python

   import numpy as np
   from sklearn.linear_model import LinearRegression
   from mmer import MixedEffectEstimator, EnsembleMixedModel

   # Simulated Data
   n_samples = 5000
   n_groups = 50
   n_features = 5

   groups = np.random.randint(0, n_groups, size=(n_samples, 1))
   X = np.random.randn(n_samples, n_features)
   y = X.sum(axis=1, keepdims=True) + np.random.randn(n_samples, 1)

2. Train Multiple Independent Models
------------------------------------
Train a list of models independently. In this example, we use random subsets (bootstrapping) of the data to capture epistemic uncertainty.

.. code-block:: python

   n_models = 5
   fitted_models = []

   for i in range(n_models):
       # Create a bootstrapped sample
       indices = np.random.choice(n_samples, size=n_samples, replace=True)
       X_boot, y_boot, groups_boot = X[indices], y[indices], groups[indices]

       # Fit a Mixed-Effects Model
       estimator = MixedEffectEstimator(fe_model=LinearRegression(), max_iter=15)
       print(f"Fitting model {i+1}/{n_models}...")
       result = estimator.fit(X_boot, y_boot, groups_boot)
       
       fitted_models.append(result)

3. Wrap in the Ensemble Model
-----------------------------
Pass your list of fitted models to the ``EnsembleMixedModel`` to aggregate the results.

.. code-block:: python

   ensemble = EnsembleMixedModel(fitted_models)

   print(ensemble.summary())

4. Interpret Expected Means and Uncertainties
---------------------------------------------
The ensemble returns a tuple of ``(Mean, Epistemic Standard Deviation)`` for predictions and variance components.

.. code-block:: python

   mean_R, std_R = ensemble.R
   print("Expected Residual Covariance:\n", mean_R)
   print("Uncertainty (Std) of Residual Covariance:\n", std_R)

   mean_G, std_G = ensemble.G[0]
   print("Expected Random Effects Covariance for Group 1:\n", mean_G)
   print("Uncertainty (Std) of Random Effects Covariance:\n", std_G)
