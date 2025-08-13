import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import solve
from .random_effect import RandomEffect
from .residual import Residual
from .operator import VLinearOperator, ResidualPreconditioner

class MixedEffectResults:
    """
    Result class for the Multivariate Mixed Effects Regression.
    """
    def __init__(self, mixed_model, random_effects: tuple[RandomEffect], residual: Residual):
        self.fe_model = mixed_model.fe_model
        self.m = mixed_model.m
        self.n = mixed_model.n
        self.k = mixed_model.k
        self.random_slopes = mixed_model.random_slopes
        self.random_effects = random_effects
        self.residual = residual
        self.log_likelihood = mixed_model.log_likelihood
        self.track_change = mixed_model.track_change
        self._is_converged = mixed_model._is_converged

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses using the fitted fixed effects models.
        
        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
        
        Returns:
            (n_samples, M) array of predicted responses.
        """
        return self.fe_model.predict(X)
    
    def sample(self, X: np.ndarray) -> np.ndarray:
        """
        Sample responses from the predictive multivariate distribution.
        Assuming no observed levels for random effects.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
        
        Returns:
            (n_samples, M) array of sampled responses.
        """
        pass
    
    def compute_random_effects_and_residual(self,  X: np.ndarray, y: np.ndarray):
        """
        Compute residual (n x m) and random effects (m x q x o).
        """
        resid = (y - self.predict(X)).T.ravel()  # marginal residual
        V_op = VLinearOperator(self.random_effects, self.residual)
        try:
            resid_cov_inv = solve(a=self.residual.cov, b=np.eye(self.m), assume_a='pos')
            M_op = ResidualPreconditioner(resid_cov_inv, self.n, self.m)
        except Exception:
            print("Warning: Singular residual covariance. If the fixed-effects model absorbs nearly all degrees of freedom, residual variance may vanish, leading to singularity.")
            M_op = None

        prec_resid, info = cg(A=V_op, b=resid, M=M_op)
        if info != 0:
            print(f"Warning: CG solver (V⁻¹(y-fx)) did not converge. Info={info}")

        total_re = np.zeros(self.m * self.n)
        mu = []
        for re in self.random_effects:
            mu.append(re.compute_mu(prec_resid))
            total_re += re.map_mu(mu[-1])

        resid -= total_re  # unexplained residual

        return mu, resid

    def summary(self):
        """
        Display a summary of the fitted multivariate mixed effects model.
        """
        # Print summary statistics
        indent0 = ""
        indent1 = "   "
        indent2 = "       "

        print("\n" + indent0 + "Multivariate Mixed Effects Model Summary")
        print("=" * 50)
        print(indent1 + f"FE Model: {type(self.fe_model).__name__}")
        print(indent1 + f"Iterations: {len(self.log_likelihood)}")
        print(indent1 + f"Converged: {self._is_converged}")
        print(indent1 + f"Log-Likelihood: {self.log_likelihood[-1]:.2f}")
        print(indent1 + f"No. Observations: {self.n}")
        print(indent1 + f"No. Response Variables: {self.m}")
        print(indent1 + f"No. Grouping Variables: {self.k}")
        print("-" * 50)
        print(indent1 + f"Residual (Unexplained) Variances")
        print(indent2 + "{:<10} {:>10}".format("Response", "Variance"))
        for m in range(self.m):
            print(indent2 + "{:<10} {:>10.4f}".format(m + 1, self.residual.cov[m, m]))
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        print(indent2 + "{:<8} {:<10} {:<15} {:>10}".format("Group", "Response", "Random Effect", "Variance"))
        for k in range(self.k):
            for i in range(self.m):
                for j in range(self.random_effects[k].q):
                    idx = i * self.random_effects[k].q + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    var = self.random_effects[k].cov[idx, idx]
                    print(indent2 + "{:<8} {:<10} {:<15} {:>10.4f}".format(k + 1, i + 1, effect_name, var))
        print("\n")