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
        max_iter: Maximum number of EM iterations (default: 10).
        mll_tol: Marginal log-likelihood convergence tolerance  (default: 0.0001).
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 10, mll_tol: float = 0.0001):
        self.fem = fixed_effects_model
        self.max_iter = max_iter
        self.mll_tol = mll_tol
    
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
        self.phi = np.eye(self.M)    # Residual covariance matrix (M x M)
        self.tau = {}                # Random effect covariance matrix across Responses and Effect types per group (M.q_k x M.q_k)
        self.mu = {}                 # Conditional random effects mean
        self.sigma = {}              # Conditional random effects covariance
        self.Z = {}                  # Random effects design matrices
        self.o = {}                  # Number of levels per group
        self.q = {}                  # Number of effect types per group
        self.mll_evol = []           # Track marginal log-likelihood
        ZTZ = {}                     # Precomputed Z_k.T @ Z_k
        IM_Z = {}                    # Precomputed Kronecker product of identity and Z_k
        self.eps = np.zeros((self.n, self.M))  # Residuals
        for k in range(self.K):
            rsc_k = X[:, self.rscol[k]] if (self.rscol is not None and self.rscol[k] is not None) else None
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
    
    def fit_fixed_effects(self, X, y, zb_sum):
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
    
    def update_phi(self, ZTZ):
        """
        Update the residual covariance matrix phi based on the residuals.
        """
        S = self.eps.T @ self.eps
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
        self._converged = False
        pbar = tqdm(range(1, self.max_iter + 1), desc="MERM Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            zb_sum = self.get_zb_sum(IM_Z)
            fX = self.fit_fixed_effects(X, y, zb_sum)
            eps_marginal = y - fX
            splu, D = self.get_splu_D(IM_Z)
            self.E_step(splu, D, eps_marginal.ravel(order='F'), IM_Z)

            zb_sum = self.get_zb_sum(IM_Z)
            np.subtract(eps_marginal, zb_sum, out=self.eps)
            self.update_phi(ZTZ)
            self.update_tau()
            mll = self.get_mll(eps_marginal.ravel(order='F'), IM_Z)
            self.mll_evol.append(mll)

            if iter_ > 1 and abs((self.mll_evol[-1] - self.mll_evol[-2]) / self.mll_evol[-2]) < self.mll_tol:
                pbar.set_description("MERM Converged")
                self._converged = True
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
            Z_k, q_k, o_k = self.design_Z(groups[:, k], X[:, self.rscol[k]] if (self.rscol is not None and self.rscol[k] is not None) else None)
            D_k = sparse.kron(self.tau[k], sparse.eye(o_k, format='csr'))
            IM_Z_k = sparse.kron(sparse.eye(self.M, format='csr'), Z_k)
            V += IM_Z_k @ D_k @ IM_Z_k.T

        # Sample from multivariate normal
        y_sampled = np.random.multivariate_normal(fX.ravel(order='F'), V.toarray())
        return y_sampled.reshape((n, self.M), order='F')
    
    def design_Z(self, group: np.ndarray, random_slope_covariates: np.ndarray = None):
        """
        Construct random effects design matrix for a grouping factor.
        
        Parameters:
            group: (n_samples,) array of group levels.
            random_slope_covariates: (n_samples, q) array for random slopes (optional).
        
        Returns:
            Z_k: Sparse matrix (n_samples, o_k * q_k).
            q_k: Number of random effects per level.
            o_k: Number of unique levels.
        """
        levels, level_indices = np.unique(group, return_inverse=True)
        n = group.shape[0]
        o = len(levels)
        q = 1 if random_slope_covariates is None else 1 + random_slope_covariates.shape[1]
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
            if random_slope_covariates is not None:
                for rs_idx in range(random_slope_covariates.shape[1]):
                    idx = base_idx + (rs_idx + 1)
                    rows[idx] = i
                    cols[idx] = (rs_idx + 1) * o + j
                    data[idx] = random_slope_covariates[i, rs_idx]
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, q * o)), q, o

    def summary(self, show_plot: bool = True):
        """
        Display a summary of the fitted multivariate mixed effects model, including:
        - Convergence statistics (iterations, marginal log-likelihood).
        - Variance components (residual and random effects variances).
        - Plots: Marginal log-likelihood, covariance heatmaps (phi, tau_k), residual histograms,
        and random effects histograms per group.
        """
        if not self.mll_evol:
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
        print(indent1 + f"Iterations: {len(self.mll_evol)}")
        print(indent1 + f"Converged: {self._converged}")
        print(indent1 + f"Marginal Log-Likelihood: {self.mll_evol[-1]:.2f}")
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

        if show_plot:
            with style():
                # 1. Marginal Log-Likelihood
                _cm = 1 / 2.54
                plt.figure(figsize=(7*_cm, 7*_cm))
                plt.plot(range(1, len(self.mll_evol) + 1), self.mll_evol, marker='o')
                plt.title("Marginal Log-Likelihood")
                plt.xlabel("Iteration")
                plt.ylabel("MLL")
                plt.grid(True, which='major', linewidth=0.15, linestyle='--')
                plt.text(0.95, 0.95, f'MLL = {self.mll_evol[-1]:.4f}', transform=plt.gca().transAxes, va='top', ha='right')
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