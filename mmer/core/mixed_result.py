import numpy as np
from .mixed_effect import MixedEffectRegressor

class MixedEffectResults:
    """
    Result container for a fitted MMER model.

    Provides access to the fitted model state (coefficients, covariances) and 
    inference methods.

    Attributes
    ----------
    model : MixedEffectRegressor
        The source model instance.
    fe_model : RegressorMixin
        Fitted fixed effects model.
    m : int
        Number of output responses.
    k : int
        Number of grouping factors.
    random_effect_terms : list of RandomEffectTerm
        Learned random effect states.
    residual_term : ResidualTerm
        Learned residual state.
    log_likelihood : list
        History of log-likelihood values during training.
    """
    def __init__(self, mixed_model: MixedEffectRegressor):
        self.model = mixed_model
        
        # Expose convenient attributes
        self.fe_model = mixed_model.fe_model
        self.m = mixed_model.m
        self.k = mixed_model.k
        self.random_effect_terms = mixed_model.random_effect_terms
        self.residual_term = mixed_model.residual_term
        self.log_likelihood = mixed_model.log_likelihood
        self.is_converged = mixed_model._is_converged
        self.best_log_likelihood = mixed_model._best_log_likelihood

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using fixed effects only.

        Parameters
        ----------
        X : np.ndarray
            Features, shape (n, p).

        Returns
        -------
        np.ndarray
            Predicted values, shape (n, m).
        """
        return self.model.predict(X)
    
    def compute_random_effects(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Estimate posterior random effects for new or existing data.

        Parameters
        ----------
        X : np.ndarray
            Features, shape (n, p).
        y : np.ndarray
            Targets, shape (n, m).
        groups : np.ndarray
            Grouping factors, shape (n, k).

        Returns
        -------
        mean_random_effects : tuple of list of np.ndarray
            Posterior means of random effects.
        residuals : np.ndarray
            Estimated residuals after accounting for fixed and random effects.
        """
        return self.model.compute_random_effects(X, y, groups)

    def summary(self):
        """
        Display a summary of the fitted multivariate mixed effects model.
        """
        indent0 = ""
        indent1 = "   "
        indent2 = "       "

        print("\n" + indent0 + "Multivariate Mixed Effects Model Summary")
        print("=" * 50)
        print(indent1 + f"FE Model: {type(self.fe_model).__name__}")
        print(indent1 + f"Iterations: {len(self.log_likelihood)}")
        print(indent1 + f"Converged: {self.is_converged}")
        print(indent1 + f"Log-Likelihood: {self.best_log_likelihood:.3f}")
        print(indent1 + f"No. Outputs: {self.m}")
        print(indent1 + f"No. Grouping Factors: {self.k}")
        print("-" * 50)
        print(indent1 + f"Unexplained Residual Variances")
        print(indent2 + "{:<10} {:>10}".format("Response", "Variance"))
        for m in range(self.m):
            print(indent2 + "{:<10} {:>10.4f}".format(m + 1, self.residual_term.cov[m, m]))
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        print(indent2 + "{:<8} {:<10} {:<15} {:>10}".format("Group", "Response", "Random Effect", "Variance"))
        
        for k, term in enumerate(self.random_effect_terms):
            # q is term.q (1 + number of slopes)
            # term.cov is (m*q, m*q)
            q = term.q
            for i in range(self.m):
                for j in range(q):
                    idx = i * q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    # Access diagonal element
                    var = term.cov[idx, idx]
                    print(indent2 + "{:<8} {:<10} {:<15} {:>10.4f}".format(k + 1, i + 1, effect_name, var))
        print("\n")