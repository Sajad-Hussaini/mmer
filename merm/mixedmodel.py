import numpy as np
from scipy.stats import probplot
from sklearn.base import RegressorMixin, clone
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from .style import style
from . import utils

class MERM:
    """
    Multivariate Mixed Effects Regression Model.
    It supports multiple responses, any fixed effects model, multiple random effects, and multiple grouping factors.
    Parameters:
        fixed_effects_model: A scikit-learn regressor or list of regressors for fixed effects.
        max_iter: Maximum number iterations (default: 50).
        tol: Log-likelihood convergence tolerance  (default: 1e-4).
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 50, tol: float = 1e-4):
        self.fem = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.logL = []
        self._converged = False
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_cols: dict):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        self.n_obs, self.n_res = y.shape
        self.n_groups = groups.shape[1]
        self.slope_cols = random_slope_cols

        z_matrices, self.n_effect, self.n_level = utils.random_effect_design_matrices(X, groups, self.n_groups, self.slope_cols)
        Im_Z = utils.block_diag_design_matrices(z_matrices, self.n_res)
        ZtZ = utils.crossprod_design_matrices(z_matrices)

        self.phi = np.eye(self.n_res)
        self.tau = {k: np.eye(self.n_res * effect_k) for k, effect_k in self.n_effect.items()}
        
        self.fem_list = [clone(self.fem) for _ in range(self.n_res)] if not isinstance(self.fem, list) else self.fem
        if len(self.fem_list) != self.n_res:
            raise ValueError(f"Expected {self.n_res} fixed effects models, got {len(self.fem_list)}")
        
        fX, eps_marginal = self.fit_fixed_effect(X, y, 0.0)
        return eps_marginal, Im_Z, ZtZ
    
    def fit_fixed_effect(self, X, y, effect_sum):
        """
        Fit the fixed effects models to the adjusted response variables.
        """
        y_adj = y - effect_sum
        fX = np.zeros_like(y_adj)
        for m in range(self.n_res):
            fX[:, m] = self.fem_list[m].fit(X, y_adj[:, m]).predict(X)
        eps_marginal = y - fX
        return fX, eps_marginal

    def e_step(self, eps_marginal, Im_Z):
        """
        Perform the E-step of the EM algorithm to compute the conditional expectation and covariance of the random effects.
        """
        splu, D = utils.splu_decomposition(self.phi, self.tau, self.n_obs, self.n_level, Im_Z)
        V_inv_eps = splu.solve(eps_marginal.ravel(order='F'))
        ll = utils.compute_logL(eps_marginal, V_inv_eps, splu, self.n_res, self.n_obs)
        mu = utils.compute_mu(V_inv_eps, D, Im_Z)
        sigma = utils.compute_sigma(D, splu, Im_Z)
        return mu, sigma, ll
    
    def m_step(self, X, y, mu, sigma, Im_Z, ZtZ):
        """
        Perform the E-step of the EM algorithm to update the fixed effects functions, residual, and random effects covariance matrices.
        """
        effect_sum = utils.sum_random_effect(mu, Im_Z, self.n_res, self.n_obs)
        _, eps_marginal = self.fit_fixed_effect(X, y, effect_sum)
        eps = eps_marginal - effect_sum
        self.eps = eps
        self.compute_residual_covariance(sigma, eps, ZtZ)
        self.compute_random_effect_covariance(mu, sigma)
        return eps_marginal
    
    def compute_residual_covariance(self, sigma, eps, ZtZ):
        """
        the residual covariance matrix phi.
        """
        S = eps.T @ eps
        T = np.zeros((self.n_res, self.n_res))
        for m1 in range(self.n_res):
            for m2 in range(self.n_res):
                trace_sum = 0.0
                for k in range(self.n_groups):
                    o_k, q_k = self.n_level[k], self.n_effect[k]
                    idx1 = slice(m1 * q_k * o_k, (m1 + 1) * q_k * o_k)
                    idx2 = slice(m2 * q_k * o_k, (m2 + 1) * q_k * o_k)
                    sigma_k_block = sigma[k][idx1, idx2]
                    trace_sum += (ZtZ[k] @ sigma_k_block).trace()
                T[m1, m2] = trace_sum
        self.phi = (S + T) / self.n_obs + 1e-6 * np.eye(self.n_res)

    def compute_random_effect_covariance(self, mu, sigma):
        """
        Update the random effects covariance matrix tau.
        """
        for k in range(self.n_groups):
            o_k, q_k = self.n_level[k], self.n_effect[k]
            mu_k = mu[k]
            sigma_k = sigma[k]
            sum_tau = np.zeros((self.n_res * q_k, self.n_res * q_k))
            for j in range(o_k):
                indices = []  # Indices for level j across all responses and effect types
                for m in range(self.n_res):
                    for q in range(q_k):
                        idx = m * q_k * o_k + q * o_k + j
                        indices.append(idx)
                mu_k_j = mu_k[indices]
                sigma_k_block = sigma_k[np.ix_(indices, indices)]
                sum_tau += np.outer(mu_k_j, mu_k_j) + sigma_k_block
            self.tau[k] = sum_tau / o_k + 1e-6 * np.eye(self.n_res * q_k)            

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_cols: dict = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slope_cols: List of (n_samples, q_k) arrays for random slopes per group (optional).

        Returns:
            Self (fitted model).
        """
        eps_marginal, Im_Z, ZtZ = self._prepare_data(X, y, groups, random_slope_cols)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Model Fitting", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            mu, sigma, ll = self.e_step(eps_marginal, Im_Z)
            self.mu = mu
            self.logL.append(ll)
            if iter_ > 2 and abs((self.logL[-1] - self.logL[-2]) / self.logL[-2]) < self.tol:
                pbar.set_description("Model Converged")
                self._converged = True
                break
            eps_marginal = self.m_step(X, y, mu, sigma, Im_Z, ZtZ)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses using the fitted fixed effects models.
        
        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
        
        Returns:
            (n_samples, M) array of predicted responses.
        """
        if not self.fem_list:
            raise ValueError("Model must be fitted before prediction.")
        n = X.shape[0]
        fX = np.zeros((n, self.n_res))
        for m in range(self.n_res):
            fX[:, m] = self.fem_list[m].predict(X)
        return fX
    
    def sample(self, X: np.ndarray) -> np.ndarray:
        """
        Sample responses from the predictive multivariate distribution.
        
        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
        
        Returns:
            (n_samples, M) array of sampled responses.
        """
        fX = self.predict(X)
        n = X.shape[0]
        y_sampled = np.zeros_like(fX)
        groups = np.zeros((n, self.n_groups), dtype=int)
        for i in range(n):
            z_matrices, n_effect, n_level = utils.random_effect_design_matrices(X[i:i+1], groups[i:i+1], self.n_groups, self.slope_cols)
            Im_Z = utils.block_diag_design_matrices(z_matrices, self.n_res)
            V_i, _ = utils.marginal_covariance(self.phi, self.tau, 1, n_level, Im_Z)
            y_sampled[i] = np.random.multivariate_normal(fX[i], V_i)
        return y_sampled
    
    def get_random_effect_residual(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_cols: dict):
        """
        Compute residuals and random effects for diagnostics.
        Returns:
            dict with 'residuals' and 'random_effects' (mu).
        """
        # Rebuild design matrices and metadata
        n_obs, n_res = y.shape
        n_groups = groups.shape[1]
        z_matrices, n_effect, n_level = utils.random_effect_design_matrices(X, groups, n_groups, random_slope_cols)
        Im_Z = utils.block_diag_design_matrices(z_matrices, n_res)
        eps_marginal = y - self.predict(X)
        splu, D = utils.splu_decomposition(self.phi, self.tau, n_obs, n_level, Im_Z)
        V_inv_eps = splu.solve(eps_marginal.ravel(order='F'))
        mu = utils.compute_mu(V_inv_eps, D, Im_Z)
        effect_sum = utils.sum_random_effect(mu, Im_Z, n_res, n_obs)
        eps = eps_marginal - effect_sum
        return mu, eps

    def summary(self, random_effect: np.ndarray = None, residual: np.ndarray = None, show_plot: bool = True):
        """
        Display a summary of the fitted multivariate mixed effects model, including:
        - Convergence statistics (iterations, marginal log-likelihood).
        - Variance components (residual and random effects variances).
        - Plots: Marginal log-likelihood, covariance heatmaps (phi, tau_k), residual histograms,
        and random effects histograms per group.
        """
        if not self.fem_list:
            raise ValueError("Model must be fitted before calling summary.")

        # Compute correlation matrices
        corr_phi = utils.cov_to_corr(self.phi)
        corr_tau = {k: utils.cov_to_corr(self.tau[k]) for k in self.tau}

        # Print summary statistics
        indent0 = ""
        indent1 = "   "
        indent2 = "       "
        indent3 = "         "
        indent4 = "            "

        print("\n" + indent0 + "Multivariate Mixed Effects Model Summary")
        print("=" * 50)
        print(indent1 + f"FE Model: {type(self.fem_list[0]).__name__}")
        print(indent1 + f"Iterations: {len(self.logL)}")
        print(indent1 + f"Converged: {self._converged}")
        print(indent1 + f"Log-Likelihood: {self.logL[-1]:.2f}")
        print(indent1 + f"No. Observations: {self.n_obs}")
        print(indent1 + f"No. Response Variables: {self.n_res}")
        print(indent1 + f"No. Grouping Variables: {self.n_groups}")
        print("-" * 50)
        print(indent1 + f"Residual (Unexplained) Variances")
        for m in range(self.n_res):
            print(indent4 + f"Response {m+1}: {self.phi[m, m]:.4f}")
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        for k in range(self.n_groups):
            print(indent2 + f"Group {k+1}:")
            for i in range(self.n_res):
                print(indent3 + f"Response {i+1}:")
                for j in range(self.n_effect[k]):
                    idx = i * self.n_effect[k] + j
                    effect_name = "intercept" if j == 0 else f"slope{j}"
                    var = self.tau[k][idx, idx]
                    print(indent4 + f"{effect_name}: {var:.4f}")
        print("\n")

        if show_plot:
            with style():
                # 1. Marginal Log-Likelihood
                _cm = 1 / 2.54
                plt.figure(figsize=(7*_cm, 7*_cm))
                plt.plot(range(1, len(self.logL) + 1), self.logL, marker='o')
                plt.title("Log-Likelihood")
                plt.xlabel("Iteration")
                plt.ylabel("LogL")
                plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                plt.text(0.95, 0.95, f'LogL = {self.logL[-1]:.4f}', transform=plt.gca().transAxes, va='top', ha='right')
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.show(block=False)

                # 2. Residual Correlation Heatmap
                phi_dim = self.phi.shape[0]
                if phi_dim > 1:
                    plt.figure(figsize=((phi_dim + 5)*_cm, (phi_dim + 5)*_cm))
                    labels = [f"R{m+1}" for m in range(self.n_res)]
                    sns.heatmap(corr_phi, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                                xticklabels= labels, yticklabels=labels)
                    plt.title(r"Residual Correlation ($\phi$)")
                    plt.show()

                    plt.figure(figsize=((phi_dim + 5)*_cm, (phi_dim + 5)*_cm))
                    sns.heatmap(self.phi, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                                xticklabels= labels, yticklabels=labels)
                    plt.title(r"Residual Covariance ($\phi$)")
                    plt.show()

                # 3. Random Effects Correlation Heatmaps
                for k in range(self.n_groups):
                    tau_dim = self.tau[k].shape[0]
                    if tau_dim > 1:
                        labels = [f"R{m+1}-{'I' if q == 0 else f'S{q+1}'}" for m in range(self.n_res) for q in range(self.n_effect[k])]
                        plt.figure(figsize=((tau_dim + 5)*_cm, (tau_dim + 5)*_cm))
                        sns.heatmap(corr_tau[k], annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                                    xticklabels=labels, yticklabels=labels)
                        plt.title(fr"Random Effects Correlation ($\tau$) for Group {k+1}")
                        plt.show()

                        plt.figure(figsize=((tau_dim + 5)*_cm, (tau_dim + 5)*_cm))
                        sns.heatmap(self.tau[k], annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                                    xticklabels=labels, yticklabels=labels)
                        plt.title(fr"Random Effects Covaraince ($\tau$) for Group {k+1}")
                        plt.show()

                # 4. Residual Histograms
                if residual is not None:
                    for m in range(self.n_res):
                        plt.figure(figsize=(7*_cm, 7*_cm))
                        plt.hist(residual[:, m], bins='auto', edgecolor='black')
                        plt.title(f"Response {m+1} Residuals")
                        plt.xlabel("Residual")
                        plt.ylabel("Frequency")
                        plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                        plt.show()

                        plt.figure(figsize=(7*_cm, 7*_cm))
                        probplot(residual[:, m], dist="norm", plot=plt)
                        plt.title(f"Response {m+1} Residuals")
                        plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                        plt.show()

                # 5. Random Effects Histograms
                if random_effect is not None:
                    for k in range(self.n_groups):
                        mu_k = random_effect[k].reshape(self.n_res, self.n_effect[k], self.n_level[k])
                        for m in range(self.n_res):
                            for j in range(self.n_effect[k]):
                                plt.figure(figsize=(7*_cm, 7*_cm))
                                plt.hist(mu_k[m, j, :], bins='auto', edgecolor='black')
                                effect_name = "Intercept" if j == 0 else f"Slope {j}"
                                plt.title(f"Group {k+1} Response {m+1} Random {effect_name}")
                                plt.xlabel("Random Effect")
                                plt.ylabel("Frequency")
                                plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                                plt.show()

                                plt.figure(figsize=(7*_cm, 7*_cm))
                                probplot(mu_k[m, j, :], dist="norm", plot=plt)
                                effect_name = "Intercept" if j == 0 else f"Slope {j}"
                                plt.title(f"Group {k+1} Response {m+1} Random {effect_name}")
                                plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                                plt.show()
    def plot_residuals_vs_fitted(self, X, y, groups, random_slope_cols):
        mu, eps = self.get_random_effect_residual(X, y, groups, random_slope_cols)
        fX = self.predict(X)
        with style():
            for m in range(self.n_res):
                plt.figure(figsize=(7*_cm, 7*_cm))
                plt.scatter(fX[:, m], eps[:, m], alpha=0.5, edgecolor='black')
                plt.axhline(0, color='red', linestyle='--', linewidth=0.5)
                plt.title(f"Response {m+1}: Residuals vs Fitted")
                plt.xlabel("Fitted Values")
                plt.ylabel("Residuals")
                plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                plt.show()