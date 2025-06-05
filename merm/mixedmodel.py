import numpy as np
import scipy.sparse as sparse
from sklearn.base import RegressorMixin, clone
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from .style import style
from . import utils

class MERM:
    """
    Multivariate Mixed Effects Regression Model using Expectation-Maximization.
    It supports multiple responses, any fixed effects model, multiple random effects, and multiple grouping factors.
    Parameters:
        fixed_effects_model: A scikit-learn regressor or list of regressors for fixed effects.
        max_iter: Maximum number of EM iterations (default: 50).
        mll_tol: Marginal log-likelihood convergence tolerance  (default: 1e-4).
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 50, tol: float = 1e-4):
        self.fem = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.logL = []
        self._converged = False
    
    def _initialization(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_column: list = None):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        # Initialize fixed effects models
        self.fem_list = [clone(self.fem) for _ in range(self.M)] if not isinstance(self.fem, list) else self.fem
        if len(self.fem_list) != self.M:
            raise ValueError(f"Expected {self.M} fixed effects models, got {len(self.fem_list)}")
        
        # State attributes
        self.n, self.M = y.shape
        self.K = groups.shape[1]
        self.rscol = random_slope_column
        self.phi = np.eye(self.M)    # Residual covariance matrix (M x M)
        self.tau = {}                # Random effect covariance matrix across Responses and Effect types per group (M.q_k x M.q_k)
        self.o = {}                  # Number of levels per group
        self.q = {}                  # Number of effect types per group
        # Local cached attributes for fitting
        mu = {}                           # Conditional random effects mean (i.e., BLUP)
        sigma = {}                        # Conditional random effects covariance
        IM_Z = {}
        ZTZ = {}

        for k in range(self.K):
            rsc_k = X[:, self.rscol[k]] if (self.rscol is not None and self.rscol[k] is not None) else None
            Z_k, self.q[k], self.o[k] = utils.design_Z(groups[:, k], rsc_k)
            ZTZ[k] = Z_k.T @ Z_k
            IM_Z[k] = utils.IM_kron_Z(self.M, Z_k)
            self.tau[k] = np.eye(self.M * self.q[k])
            mu[k] = np.zeros(self.M * self.q[k] * self.o[k])
            sigma[k] = sparse.eye_array(self.M * self.q[k] * self.o[k], format='csr')
        fX, eps_marginal = self.fit_fixed_effect(X, y, 0.0)
        return mu, sigma, eps_marginal, IM_Z, ZTZ
    
    def map_random_effect(self, mu, IM_Z, k):
        return (IM_Z[k] @ mu[k]).reshape((self.n, self.M), order='F')

    def sum_random_effect(self, mu, IM_Z):
        """
        Compute the sum of random effects contributions for all groups.
        """
        zb_sum = np.zeros((self.n, self.M))
        for k in range(self.K):
            zb_sum += self.map_random_effect(mu, IM_Z, k)
        return zb_sum
    
    def fit_fixed_effect(self, X, y, zb_sum):
        """
        Fit the fixed effects models to the adjusted response variables.
        """
        y_adj = y - zb_sum
        fX = np.zeros((self.n, self.M))
        for m in range(self.M):
            fX[:, m] = self.fem_list[m].fit(X, y_adj[:, m]).predict(X)
        eps_marginal = y - fX
        return fX, eps_marginal
    
    def compute_splu_covariance(self, IM_Z):
        """
        Compute the sparse LU decomposition of the marginal covariance matrix V and the random effects covariance matrices D.
        """
        D = {}
        V = sparse.kron(self.phi, sparse.eye_array(self.n, format='csr'), format='csr')
        for k in range(self.K):
            D[k] = sparse.kron(self.tau[k], sparse.eye_array(self.o[k], format='csr'), format='csr')
            V += IM_Z[k] @ D[k] @ IM_Z[k].T
        return sparse.linalg.splu(V.tocsc()), D
    
    def e_step(self, mu, sigma, eps_marginal, IM_Z):
        """
        Perform the E-step of the EM algorithm to compute the conditional expectation and covariance of the random effects.
        """
        splu, D = self.compute_splu_covariance(IM_Z)
        V_inv_eps = splu.solve(eps_marginal.ravel(order='F'))
        ll = self.compute_logL(eps_marginal, V_inv_eps, splu)
        for k in range(self.K):
            mu[k] = D[k] @ IM_Z[k].T @ V_inv_eps
            IM_Zk_Dk = IM_Z[k] @ D[k]
            V_inv_IM_Zk_Dk = sparse.csr_array(splu.solve(IM_Zk_Dk.toarray()))
            sigma[k] = D[k] - D[k] @ IM_Z[k].T @ V_inv_IM_Zk_Dk
        return mu, sigma, ll
    
    def m_step(self, X, y, mu, sigma, IM_Z, ZTZ):
        """
        Perform the E-step of the EM algorithm to compute the conditional expectation and covariance of the random effects.
        """
        zb_sum = self.sum_random_effect(mu, IM_Z)
        fX, eps_marginal = self.fit_fixed_effect(X, y, zb_sum)
        eps = eps_marginal - zb_sum
        self.compute_residual_covariance(sigma, eps, ZTZ)
        self.compute_random_effect_covariance(mu, sigma)
        return eps_marginal
    
    def compute_residual_covariance(self, sigma, eps, ZTZ):
        """
        Update the residual covariance matrix phi.
        """
        S = eps.T @ eps
        T = np.zeros((self.M, self.M))
        for m1 in range(self.M):
            for m2 in range(self.M):
                trace_sum = 0.0
                for k in range(self.K):
                    o_k, q_k = self.o[k], self.q[k]
                    idx1 = slice(m1 * q_k * o_k, (m1 + 1) * q_k * o_k)
                    idx2 = slice(m2 * q_k * o_k, (m2 + 1) * q_k * o_k)
                    sigma_k_block = sigma[k][idx1, idx2]
                    trace_sum += (ZTZ[k] @ sigma_k_block).trace()
                T[m1, m2] = trace_sum
        self.phi = (S + T) / self.n + 1e-6 * np.eye(self.M)

    def compute_random_effect_covariance(self, mu, sigma):
        """
        Update the random effects covariance matrix tau.
        """
        for k in range(self.K):
            o_k, q_k = self.o[k], self.q[k]
            mu_k = mu[k]
            sigma_k = sigma[k]
            sum_tau = np.zeros((self.M * q_k, self.M * q_k))
            for j in range(o_k):
                indices = []  # Indices for level j across all responses and effect types
                for m in range(self.M):
                    for q in range(q_k):
                        idx = m * q_k * o_k + q * o_k + j
                        indices.append(idx)
                mu_k_j = mu_k[indices]
                sigma_k_block = sigma_k[np.ix_(indices, indices)]
                sum_tau += np.outer(mu_k_j, mu_k_j) + sigma_k_block
            self.tau[k] = sum_tau / o_k + 1e-6 * np.eye(self.M * q_k)
    
    def compute_logL(self, eps_marginal, V_inv_eps, splu):
        """
        Compute the marginal log-likelihood of the model given the current parameters.
        """
        eps_marginal_stacked = eps_marginal.ravel(order='F')
        log_det_V = np.sum(np.log(np.abs(splu.U.diagonal())))
        ll = -(self.M * self.n * np.log(2 * np.pi) + log_det_V + eps_marginal_stacked.T @ V_inv_eps) / 2
        return ll
            

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_column: list = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slope_column: List of (n_samples, q_k) arrays for random slopes per group (optional).

        Returns:
            Self (fitted model).
        """
        mu, sigma, eps_marginal, IM_Z, ZTZ = self._initialization(X, y, groups, random_slope_column)
        pbar = tqdm(range(1, self.max_iter + 1), desc="Model Fitting", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            mu, sigma, ll = self.e_step(mu, sigma, eps_marginal, IM_Z)
            self.logL.append(ll)
            if iter_ > 2 and abs((self.logL[-1] - self.logL[-2]) / self.logL[-2]) < self.tol:
                pbar.set_description("Model Converged")
                self._converged = True
                break
            eps_marginal = self.m_step(X, y, mu, sigma, IM_Z, ZTZ)
        return self
    
    def predict(self, X: np.ndarray, sample: bool = False) -> np.ndarray:
        """
        Predict responses using the fitted fixed effects models.
        
        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            sample: If True, sample from multivariate normal with covariance V; else return mean.
        
        Returns:
            (n_samples, M) array of predicted responses (mean or sampled).
        """
        if not self.fem_list:
            raise ValueError("Model must be fitted before prediction.")
        n = X.shape[0]
        fX = np.zeros((n, self.M))
        for m in range(self.M):
            fX[:, m] = self.fem_list[m].predict(X)
        if not sample:
            return fX

        y_sampled = self._sample(n, X, fX)
        return y_sampled
    
    def _sample(self, n, X, fX):
        """
        Sample responses from the fitted model using the multivariate normal distribution.
        """
        y_sampled = np.zeros((n, self.M))
        groups = np.zeros((n, self.K), dtype=int)
        for i in range(n):
            V_i = self.phi.copy()
            for k in range(self.K):
                Z_k, _, _ = utils.design_Z(groups[i], X[i, self.rscol[k]] if (self.rscol is not None and self.rscol[k] is not None) else None)

                IM_Z_k = np.kron(np.eye(self.M), Z_k)
                V_i += IM_Z_k @ self.tau[k] @ IM_Z_k.T

            y_sampled[i] = np.random.multivariate_normal(fX[i], V_i)
        return y_sampled

    def summary(self, show_plot: bool = True):
        """
        Display a summary of the fitted multivariate mixed effects model, including:
        - Convergence statistics (iterations, marginal log-likelihood).
        - Variance components (residual and random effects variances).
        - Plots: Marginal log-likelihood, covariance heatmaps (phi, tau_k), residual histograms,
        and random effects histograms per group.
        """
        if not self.logL:
            raise ValueError("Model must be fitted before calling summary.")

        # Compute correlation matrices
        corr_phi = utils.cov_to_corr(self.phi)
        corr_tau = {k: utils.cov_to_corr(self.tau[k]) for k in range(self.K)}

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
        print(indent1 + f"Marginal Log-Likelihood: {self.logL[-1]:.2f}")
        print(indent1 + f"No. Observations: {self.n}")
        print(indent1 + f"No. Response Variables: {self.M}")
        print(indent1 + f"No. Grouping Variables: {self.K}")
        print("-" * 50)
        print(indent1 + f"Residual (Unexplained) Variances")
        for m in range(self.M):
            print(indent4 + f"Response {m+1}: {self.phi[m, m]:.4f}")
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        for k in range(self.K):
            print(indent2 + f"Group {k+1}:")
            for i in range(self.M):
                print(indent3 + f"Response {i+1}:")
                for j in range(self.q[k]):
                    idx = i * self.q[k] + j
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
                plt.title("Marginal Log-Likelihood")
                plt.xlabel("Iteration")
                plt.ylabel("MLL")
                plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                plt.text(0.95, 0.95, f'MLL = {self.logL[-1]:.4f}', transform=plt.gca().transAxes, va='top', ha='right')
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.show(block=False)

                # 2. Residual Correlation Heatmap (phi)
                phi_dim = self.phi.shape[0]
                if phi_dim > 1:
                    plt.figure(figsize=((phi_dim + 5)*_cm, (phi_dim + 5)*_cm))
                    labels = [f"R{m+1}" for m in range(self.M)]
                    sns.heatmap(corr_phi, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                                xticklabels= labels, yticklabels=labels)
                    plt.title(r"Residual Correlation ($\phi$)")
                    plt.show()

                    plt.figure(figsize=((phi_dim + 5)*_cm, (phi_dim + 5)*_cm))
                    sns.heatmap(self.phi, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                                xticklabels= labels, yticklabels=labels)
                    plt.title(r"Residual Covariance ($\phi$)")
                    plt.show()

                # 3. Random Effects Correlation Heatmaps (tau_k, up to max_plots groups)
                for k in range(self.K):
                    tau_dim = self.tau[k].shape[0]
                    if tau_dim > 1:
                        labels = [f"R{m+1}-E{q+1}" for m in range(self.M) for q in range(self.q[k])]
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

                # 4. Residual Histograms (combined if M <= max_plots, else separate)
                for m in range(self.M):
                    plt.figure(figsize=(7*_cm, 7*_cm))
                    plt.hist(self.eps[:, m], bins='auto', edgecolor='black')
                    plt.title(f"Residuals (Response {m+1})")
                    plt.xlabel("Residual")
                    plt.ylabel("Frequency")
                    plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                    plt.show()

                # 5. Random Effects Histograms (combined per group if M <= max_plots, else separate)
                for k in range(self.K):
                    mu_k = self.mu[k].reshape(self.M, self.q[k] * self.o[k], order='F')
                    for m in range(self.M):
                        plt.figure(figsize=(7*_cm, 7*_cm))
                        plt.hist(mu_k[m], bins='auto', edgecolor='black')
                        plt.title(f"Random Effects (Group {k+1}, Response {m+1})")
                        plt.xlabel("Random Effect")
                        plt.ylabel("Frequency")
                        plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                        plt.show()