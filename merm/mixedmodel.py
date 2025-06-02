import numpy as np
import scipy.sparse as sparse
from sklearn.base import RegressorMixin, clone
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from .style import style

class MERM:
    """
    Multivariate Mixed Effects Regression Model using Expectation-Maximization.
    It supports multiple responses, any fixed effects model, multiple random effects, and multiple grouping factors.
    Parameters:
        fixed_effects_model: A scikit-learn regressor or list of regressors for fixed effects.
        max_iter: Maximum number of EM iterations (default: 10).
        mll_tol: Marginal log-likelihood convergence tolerance  (default: 0.0001).
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 10, mll_tol: float = 0.0001):
        self.fem = fixed_effects_model
        self.max_iter = max_iter
        self.mll_tol = mll_tol

        self.phi = None     # Residual covariance matrix (M x M)
        self.tau = {}       # Random effect covariance matrix across Responses and Effect types per group (M.q_k x M.q_k)
        self.mu = {}        # Conditional random effects mean
        self.sigma = {}     # Conditional random effects covariance
        self.Z = {}         # Random effects design matrices
        self.o = {}         # Number of levels per group
        self.q = {}         # Number of effect types per group
        self.fem_list = []  # List of fitted fixed effects models
        self.mll_evol = []  # Track marginal log-likelihood
    
    def _initialization(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_column: list[np.ndarray] = None):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        self.n, self.M = y.shape
        self.K = groups.shape[1]
        self.rscol = random_slope_column

        # Initialize fixed effects models
        self.fem_list = [clone(self.fem) for _ in range(self.M)] if not isinstance(self.fem, list) else self.fem
        if len(self.fem_list) != self.M:
            raise ValueError(f"Expected {self.M} fixed effects models, got {len(self.fem_list)}")
        
        # Compute design matrices and initialize parameters
        ZTZ = {}       # Precomputed Z_k.T @ Z_k
        IM_Z = {}      # Precomputed Kronecker product of identity and Z_k
        self.phi = np.eye(self.M)
        for k in range(self.K):
            rsc_k = X[:, self.rscol[k]] if self.rscol is not None else None
            self.Z[k], self.q[k], self.o[k] = self.design_Z(groups[:, k], rsc_k)
            ZTZ[k] = self.Z[k].T @ self.Z[k]
            IM_Z[k] = sparse.kron(sparse.eye(self.M, format='csr'), self.Z[k])
            self.tau[k] = np.eye(self.M * self.q[k])
            self.mu[k] = np.zeros(self.M * self.q[k] * self.o[k])
            self.sigma[k] = np.eye(self.M * self.q[k] * self.o[k])
        return ZTZ, IM_Z

    def get_zb_sum(self, IM_Z):
        """
        Compute the sum of random effects contributions for all groups.
        """
        zb_sum = np.zeros((self.n, self.M))
        for k in range(self.K):
            zb_sum += (IM_Z[k] @ self.mu[k]).reshape((self.n, self.M), order='F')
        return zb_sum
    
    def fit_fixed_effects(self, X: np.ndarray, y: np.ndarray, zb_sum: np.ndarray):
        """
        Fit the fixed effects models to the adjusted response variables.
        """
        y_adj = y - zb_sum
        fX = np.zeros((self.n, self.M))
        for m in range(self.M):
            fX[:, m] = self.fem_list[m].fit(X, y_adj[:, m]).predict(X)
        return fX
    
    def get_splu_D(self, IM_Z):
        """
        Compute the sparse LU decomposition of the covariance matrix V and the random effects covariance matrices D_k.
        """
        D = {}
        V = sparse.kron(self.phi, sparse.eye(self.n, format='csr'))
        for k in range(self.K):
            D[k] = sparse.kron(self.tau[k], sparse.eye(self.o[k], format='csr'))
            V += IM_Z[k] @ D[k] @ IM_Z[k].T
        return sparse.linalg.splu(V.tocsc()), D
    
    def E_step(self, splu, D, eps_marginal_stacked, IM_Z):
        """
        Perform the E-step of the EM algorithm to compute the conditional expectation and covariance of the random effects.
        """
        V_inv_eps = splu.solve(eps_marginal_stacked)
        for k in range(self.K):
            self.mu[k] = D[k] @ IM_Z[k].T @ V_inv_eps
            IM_Zk_Dk = IM_Z[k] @ D[k]
            V_inv_IM_Zk_Dk = splu.solve(IM_Zk_Dk.toarray())
            self.sigma[k] = D[k] - D[k] @ IM_Z[k].T @ V_inv_IM_Zk_Dk
    
    def update_phi(self, eps, ZTZ):
        """
        Update the residual covariance matrix phi based on the residuals.
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
                    Sigma_k_block = self.sigma[k][idx1, idx2]
                    trace_sum += np.trace(ZTZ[k] @ Sigma_k_block)
                T[m1, m2] = trace_sum
        self.phi = (S + T) / self.n + 1e-10 * np.eye(self.M)
    
    def update_tau(self):
        """
        Update the random effects covariance tau_k based on the mu_k and sigma_k.
        """
        for k in range(self.K):
            o_k, q_k = self.o[k], self.q[k]
            mu_k = self.mu[k]
            Sigma_k = self.sigma[k]
            sum_tau = np.zeros((self.M * q_k, self.M * q_k))
            for j in range(o_k):
                indices = []  # Indices for level j across all responses and effect types
                for m in range(self.M):
                    for q in range(q_k):
                        idx = m * q_k * o_k + q * o_k + j
                        indices.append(idx)
                mu_k_ij = mu_k[indices]
                Sigma_k_ij = Sigma_k[np.ix_(indices, indices)]
                sum_tau += np.outer(mu_k_ij, mu_k_ij) + Sigma_k_ij
            self.tau[k] = sum_tau / o_k + 1e-10 * np.eye(self.M * q_k)
    
    def get_mll(self, eps_marginal_stacked, IM_Z):
        """
        Compute the marginal log-likelihood of the model given the current parameters.
        """
        splu, D = self.get_splu_D(IM_Z)
        V_inv_eps = splu.solve(eps_marginal_stacked)
        log_det_V = np.sum(np.log(np.abs(splu.U.diagonal())))
        mll = -(self.M * self.n * np.log(2 * np.pi) + log_det_V + eps_marginal_stacked.T @ V_inv_eps) / 2
        return mll
            

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_column: list[np.ndarray] = None):
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
        ZTZ, IM_Z = self._initialization(X, y, groups, random_slope_column)
        pbar = tqdm(range(1, self.max_iter + 1), desc="MERM Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            zb_sum = self.get_zb_sum(IM_Z)
            fX = self.fit_fixed_effects(X, y, zb_sum)
            eps_marginal_stacked = (y - fX).ravel(order='F')
            splu, D = self.get_splu_D(IM_Z)
            self.E_step(splu, D, eps_marginal_stacked, IM_Z)

            zb_sum = self.get_zb_sum(IM_Z)
            eps = y - fX - zb_sum
            self.update_phi(eps, ZTZ)
            self.update_tau()
            mll = self.get_mll(eps_marginal_stacked, IM_Z)
            self.mll_evol.append(mll)

            if iter_ > 1 and abs((self.mll_evol[-1] - self.mll_evol[-2]) / self.mll_evol[-2]) < self.mll_tol:
                pbar.set_description("MERM Converged")
                break
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

        # Covariance V for new X data using phi and tau_k assuming unobserved grouping
        groups = np.zeros((n, self.K), dtype=int)
        V = sparse.kron(self.phi, sparse.eye(n, format='csr'))
        for k in range(self.K):
            Z_k, q_k, o_k = self.design_Z(groups[:, k], X[:, self.rscol[k]] if self.rscol is not None else None)
            D_k = sparse.kron(self.tau[k], sparse.eye(o_k, format='csr'))
            IM_Z_k = sparse.kron(sparse.eye(self.M, format='csr'), Z_k)
            V += IM_Z_k @ D_k @ IM_Z_k.T

        # Sample from multivariate normal
        y_sampled = np.random.multivariate_normal(fX.ravel(order='F'), V.toarray())
        return y_sampled.reshape((n, self.M), order='F')
    
    def design_Z(self, group: np.ndarray, random_slope_column: np.ndarray = None):
        """
        Construct random effects design matrix for a grouping factor.
        
        Parameters:
            group: (n_samples,) array of group levels.
            random_slope_column: (n_samples, q) array for random slopes (optional).
        
        Returns:
            Z_k: Sparse matrix (n_samples, o_k * q_k).
            o_k: Number of unique levels.
            q_k: Number of random effects per level.
        """
        levels, level_indices = np.unique(group, return_inverse=True)
        n = group.shape[0]
        o = len(levels)
        q = 1 if random_slope_column is None else 1 + random_slope_column.shape[1]
        # Number of non-zero elements
        nnz = n * q
        rows = np.zeros(nnz, dtype=int)
        cols = np.zeros(nnz, dtype=int)
        data = np.zeros(nnz, dtype=float)
        for i in range(n):
            j = level_indices[i]  # level index (0 to o-1)
            base_idx = i * q      # Starting (intercept) index in sparse arrays
            # Intercept
            rows[base_idx] = i
            cols[base_idx] = j    # 0 * o + j
            data[base_idx] = 1.0
            # Slopes
            if random_slope_column is not None:
                for rs_idx in range(random_slope_column.shape[1]):
                    idx = base_idx + (rs_idx + 1)
                    rows[idx] = i
                    cols[idx] = (rs_idx + 1) * o + j
                    data[idx] = random_slope_column[i, rs_idx]
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, q * o)), q, o

    def summary(self, X, y):
        """
        Display a summary of the fitted multivariate mixed effects model, including:
        - Convergence statistics (iterations, marginal log-likelihood).
        - Variance components (residual and random effects variances).
        - Plots: Marginal log-likelihood, covariance heatmaps (phi, tau_k), residual histograms,
        and random effects histograms per group.
        """
        if not self.mll_evol:
            raise ValueError("Model must be fitted before calling summary.")
        # if not hasattr(self, 'X') or not hasattr(self, 'y'):
        #     raise ValueError("X and y must be stored in self during fit for summary.")

        # Compute correlation matrices
        phi_diag_sqrt = np.sqrt(np.diag(self.phi))
        corr_phi = self.phi / np.outer(phi_diag_sqrt, phi_diag_sqrt)
        corr_tau = {}
        for k in range(self.K):
            tau_diag_sqrt = np.sqrt(np.diag(self.tau[k]))
            corr_tau[k] = self.tau[k] / np.outer(tau_diag_sqrt, tau_diag_sqrt)

        # Compute residuals
        IM_Z = {k: sparse.kron(sparse.eye(self.M, format='csr'), self.Z[k]) for k in range(self.K)}
        zb_sum = self.get_zb_sum(IM_Z)
        fX = np.zeros((self.n, self.M))
        for m in range(self.M):
            fX[:, m] = self.fem_list[m].predict(X)
        eps = y - fX - zb_sum  # Residuals (n x M)

        # Print summary statistics
        print("Multivariate Mixed Effects Model Summary:")
        print(f"- Iterations: {len(self.mll_evol)}")
        print(f"- Final Marginal Log-Likelihood: {self.mll_evol[-1]:.2f}")
        print(f"- Residual Variances (per response):")
        for m in range(self.M):
            print(f"  Response {m+1}: {self.phi[m, m]:.4f}")
        print(f"- Random Effects Average Variances (per group):")
        for k in range(self.K):
            avg_var = np.mean(np.diag(self.tau[k]))
            print(f"  Group {k+1}: {avg_var:.4f}")

        # Plotting
        with plt.style.context('default'):  # Adjust if custom style is used
            # Create figure layout: 2 rows, 2 columns for main plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
            axes = axes.flatten()

            # 1. Marginal Log-Likelihood
            axes[0].plot(self.mll_evol, marker='o')
            axes[0].set_title("Marginal Log-Likelihood")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("MLL")
            axes[0].grid(True, which='major', linewidth=0.15, linestyle='--')
            axes[0].text(0.95, 0.95, f'MLL = {self.mll_evol[-1]:.2f}', transform=axes[0].transAxes, va='top', ha='right')

            # 2. Residual Correlation Heatmap (phi)
            sns.heatmap(corr_phi, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1],
                        xticklabels=[f"R{m+1}" for m in range(self.M)],
                        yticklabels=[f"R{m+1}" for m in range(self.M)])
            axes[1].set_title("Residual Correlation (phi)")

            # 3. Residual Histogram (first response or all if M <= 3)
            if self.M <= 3:
                for m in range(self.M):
                    axes[2].hist(eps[:, m], bins=20, alpha=0.5, label=f"Response {m+1}", edgecolor='black')
                axes[2].set_title("Residuals")
                axes[2].set_xlabel("Residual")
                axes[2].set_ylabel("Frequency")
                axes[2].legend()
            else:
                axes[2].hist(eps[:, 0], bins=20, edgecolor='black')
                axes[2].set_title("Residuals (Response 1)")
                axes[2].set_xlabel("Residual")
                axes[2].set_ylabel("Frequency")

            # 4. Random Effects Correlation Heatmap (tau[0] for first group)
            if self.K > 0:
                labels = [f"R{m+1} E{q+1}" for m in range(self.M) for q in range(self.q[0])]
                sns.heatmap(corr_tau[0], annot=len(labels) <= 6, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[3],
                            xticklabels=labels, yticklabels=labels)
                axes[3].set_title("Random Effects Correlation (tau[0])")

            # Adjust axes
            for ax in axes:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.minorticks_on()
                ax.grid(True, which='major', linewidth=0.15, linestyle='--')

            plt.show()

            # Additional tau[k] plots for other groups (if K <= 3)
            if self.K > 1 and self.K <= 3:
                for k in range(1, self.K):
                    plt.figure(figsize=(6, 5))
                    labels = [f"R{m+1} E{q+1}" for m in range(self.M) for q in range(self.q[k])]
                    sns.heatmap(corr_tau[k], annot=len(labels) <= 6, cmap='coolwarm', vmin=-1, vmax=1,
                                xticklabels=labels, yticklabels=labels)
                    plt.title(f"Random Effects Correlation (tau[{k}])")
                    plt.show()

            # Random Effects Histograms (first group, all responses if M <= 3)
            if self.K > 0:
                plt.figure(figsize=(12, 5))
                mu_k = self.mu[0].reshape(self.M, self.q[0] * self.o[0], order='F')
                for m in range(min(self.M, 3)):  # Limit to 3 responses
                    plt.hist(mu_k[m], bins=20, alpha=0.5, label=f"Response {m+1}", edgecolor='black')
                plt.title("Random Effects (Group 1)")
                plt.xlabel("Random Effect")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                plt.show()