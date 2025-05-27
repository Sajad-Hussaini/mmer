import numpy as np
import scipy.sparse as sparse
import scipy.linalg
from sklearn.base import RegressorMixin, clone
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .style import style

class MERM:
    """
    Multivariate Mixed Effects Regression Model using Expectation-Maximization.
    It supports multiple responses, any fixed effects model, multiple random effects, and multiple grouping factors.
    Parameters:
        fixed_effects_model: A scikit-learn regressor or list of regressors for fixed effects.
        max_iter: Maximum number of EM iterations (default: 10).
        mll_tol: Tolerance for convergence based on marginal log-likelihood (default: 0.001).
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: int = 10, mll_tol: float = 0.001):
        self.fem = fixed_effects_model
        self.max_iter = max_iter
        self.mll_tol = mll_tol

        self.phi = None  # Residual covariance matrix (M x M)
        self.rho = {}    # Response covariance per group (M x M)
        self.tau = {}    # Effect type covariance per group (q_k x q_k)
        self.mu = {}     # Conditional random effects mean
        self.sigma = {}  # Conditional random effects covariance
        self.Z = {}      # Random effects design matrices
        self.o = {}      # Number of levels per group
        self.q = {}      # Number of effect types per group
        self.ZTZ = {}    # Precomputed Z_k.T @ Z_k
        self.IM_Zk = {}
        self.fem_list = []  # List of fitted fixed effects models
        self.mll_evol = []  # Track marginal log-likelihood

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_covariates: list = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slope_covariates: List of (n_samples, q_k) arrays for random slopes per group (optional).

        Returns:
            Self (fitted model).
        """
        n, M = y.shape
        K = groups.shape[1]

        # Initialize fixed effects models
        if isinstance(self.fem, list):
            if len(self.fem) != M:
                raise ValueError(f"Expected {M} fixed effects models, got {len(self.fem)}")
            self.fem_list = self.fem
        else:
            self.fem_list = [clone(self.fem) for _ in range(M)]

        # Compute design matrices Z_k
        for k in range(K):
            group_k = groups[:, k]
            rsc_k = None if random_slope_covariates is None else random_slope_covariates[k]
            Z_k, o_k, q_k = self.design_Z(group_k, rsc_k)
            self.Z[k] = Z_k
            self.o[k] = o_k
            self.q[k] = q_k
            self.ZTZ[k] = Z_k.T @ Z_k  # Precompute for efficiency
            self.IM_Zk[k] = sparse.kron(sparse.eye(M, format='csr'), Z_k)

        # Initialize parameters
        self.phi = np.eye(M)
        for k in range(K):
            self.rho[k] = np.eye(M)
            self.tau[k] = np.eye(self.q[k])
            self.mu[k] = np.zeros(M * self.o[k] * self.q[k])
            self.sigma[k] = np.eye(M * self.o[k] * self.q[k])

        pbar = tqdm(range(1, self.max_iter + 1), desc="MERM Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            # Compute random effects contribution
            zb_sum = np.zeros((n, M))
            for k in range(K):
                zb_k = self.IM_Zk[k] @ self.mu[k]
                zb_sum += zb_k.reshape((M, n)).T

            # Adjust y and fit fixed effects
            y_adj = y - zb_sum
            for m in range(M):
                self.fem_list[m].fit(X, y_adj[:, m])
            fX = np.column_stack([self.fem_list[m].predict(X) for m in range(M)])

            # Compute marginal residuals
            eps = y - fX
            eps_stacked = eps.ravel(order='F')

            # Compute V sparsely
            R = sparse.kron(self.phi, sparse.eye(n, format='csr'))
            V = R.copy()
            for k in range(K):
                D_k = sparse.kron(self.rho[k], sparse.kron(sparse.eye(self.o[k], format='csr'), self.tau[k]))
                V += self.IM_Zk[k] @ D_k @ self.IM_Zk[k].T

            # Solve linear systems
            splu = sparse.linalg.splu(V.tocsc())
            V_inv_eps = splu.solve(eps_stacked)

            # E-step: Update mu_k and Sigma_k
            for k in range(K):
                D_k = sparse.kron(self.rho[k], sparse.kron(np.eye(self.o[k]), self.tau[k]))
                self.mu[k] = D_k @ self.IM_Zk[k].T @ V_inv_eps
                IM_Zk_D_k = self.IM_Zk[k] @ D_k
                V_inv_IM_Zk_D_k = splu.solve(IM_Zk_D_k.toarray())
                self.sigma[k] = D_k - D_k @ self.IM_Zk[k].T @ V_inv_IM_Zk_D_k
            
            # M-step: Update parameters
            zb_sum_stacked = zb_sum.ravel(order='F')
            y_stacked = y.ravel(order='F')
            fX_stacked = fX.ravel(order='F')
            eps_stacked = y_stacked - fX_stacked - zb_sum_stacked
            eps = eps_stacked.reshape((n, M), order='F')
            S = eps.T @ eps

            # Compute T
            T = np.zeros((M, M))
            for m1 in range(M):
                for m2 in range(M):
                    trace_sum = 0
                    for k in range(K):
                        o_k, q_k = self.o[k], self.q[k]
                        idx1 = slice(m1 * o_k * q_k, (m1 + 1) * o_k * q_k)
                        idx2 = slice(m2 * o_k * q_k, (m2 + 1) * o_k * q_k)
                        Sigma_k_block = self.sigma[k][idx1, idx2]
                        trace_sum += np.trace(self.ZTZ[k] @ Sigma_k_block)
                    T[m1, m2] = trace_sum
            self.phi = (S + T) / n + 1e-10 * np.eye(M)

            # Update rho_k and tau_k
            for k in range(K):
                o_k, q_k = self.o[k], self.q[k]
                mu_k = self.mu[k]
                Sigma_k = self.sigma[k]

                # rho_k
                sum_rho = np.zeros((M, M))
                for i in range(o_k):
                    for j in range(q_k):
                        idx = i * q_k + j
                        indices = np.array([m * o_k * q_k + idx for m in range(M)])
                        mu_k_ij = mu_k[indices]
                        Sigma_k_ij = Sigma_k[np.ix_(indices, indices)]
                        sum_rho += np.outer(mu_k_ij, mu_k_ij) + Sigma_k_ij
                self.rho[k] = sum_rho / (o_k * q_k) + 1e-10 * np.eye(M)

                # tau_k
                sum_tau = np.zeros((q_k, q_k))
                for m in range(M):
                    for i in range(o_k):
                        start = m * o_k * q_k + i * q_k
                        end = start + q_k
                        mu_k_mi = mu_k[start:end]
                        Sigma_k_mi = Sigma_k[start:end, start:end]
                        sum_tau += np.outer(mu_k_mi, mu_k_mi) + Sigma_k_mi
                self.tau[k] = sum_tau / (M * o_k) + 1e-10 * np.eye(q_k)

            # Convergence check
            log_det_V = np.sum(np.log(np.abs(splu.U.diagonal())))
            mll = -(M * n * np.log(2 * np.pi) + log_det_V + eps_stacked.T @ V_inv_eps) / 2
            self.mll_evol.append(mll)
            if iter_ > 1 and abs(self.mll_evol[-1] - self.mll_evol[-2]) < self.mll_tol:
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
        fX = np.column_stack([self.fem_list[m].predict(X) for m in range(len(self.fem_list))])
        if not sample:
            return fX
        # Construct residual covariance R = phi ⊗ I_n
        n = X.shape[0]
        M = len(self.fem_list)
        # Construct covariance V, assuming all observations in same group level
        R = sparse.kron(self.phi, sparse.eye(n, format='csc'), format='csc')
        V = R.copy()
        for k in range(len(self.rho)):
            # Assume single level: Z_k = 1_n (n x 1 vector of ones)
            Z_k = np.ones((n, 1))
            IM_Zk = sparse.kron(sparse.eye(M, format='csc'), Z_k, format='csc')
            # D_k = rho_k ⊗ tau_k (since o_k = 1)
            D_k = sparse.kron(self.rho[k], self.tau[k], format='csc')
            # Compute (I_M ⊗ Z_k) D_k (I_M ⊗ Z_k^T)
            ZkZkT = np.ones((n, n))  # Z_k Z_k^T = 1_n 1_n^T
            V += sparse.kron(D_k, ZkZkT, format='csc')

        # Sample from multivariate normal
        fX_stacked = fX.ravel(order='F')
        y_sampled = np.random.multivariate_normal(fX_stacked, V.toarray())
        return y_sampled.reshape((n, M), order='F')
    
    def design_Z(self, group: np.ndarray, random_slope_covariates: np.ndarray = None):
        """
        Construct random effects design matrix for a grouping factor.
        
        Parameters:
            group: (n_samples,) array of group levels.
            random_slope_covariates: (n_samples, q) array for random slopes (optional).
        
        Returns:
            Z_k: Sparse matrix (n_samples, o_k * q_k).
            o_k: Number of unique levels.
            q_k: Number of random effects per level.
        """
        levels = np.unique(group)
        n = group.shape[0]
        o = len(levels)
        q = 1 if random_slope_covariates is None else 1 + random_slope_covariates.shape[1]
        # Map levels to 0-based indices
        level_map = {level: idx for idx, level in enumerate(levels)}
        level_indices = np.array([level_map[level] for level in group])
        # Number of non-zero elements
        nnz = n * q
        rows = np.zeros(nnz, dtype=int)
        cols = np.zeros(nnz, dtype=int)
        data = np.zeros(nnz, dtype=float)
        for i in range(n):
            j = level_indices[i]  # Level index
            base_idx = i * q    # Starting index in sparse arrays
            base_col = j * q    # Starting column in Z
            # Intercept
            rows[base_idx] = i
            cols[base_idx] = base_col
            data[base_idx] = 1.0
            # Slopes
            if random_slope_covariates is not None:
                for rsc_idx in range(random_slope_covariates.shape[1]):
                    idx = base_idx + (rsc_idx + 1)
                    rows[idx] = i
                    cols[idx] = base_col + (rsc_idx + 1)
                    data[idx] = random_slope_covariates[i, rsc_idx]
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, o * q)), o, q

    def summary(self):
        """
        Plots of variance components and residuals and validation per iteration
        """
        with style():
            fig, axes = plt.subplots(3, 2, figsize=(12/2.54, 12/2.54), layout='constrained')
            axes = axes.flatten()

            # Generalized Log-Likelihood
            axes[0].plot(self.mll_evol, marker='o')
            axes[0].set_ylabel("MLL")
            axes[0].set_xlabel("Iteration")
            axes[0].text(0.95, 0.95, f'MLL = {self.mll_evol[-1]:.2f}', transform=axes[0].transAxes, va='top', ha='right')

            # # Validation MSE
            # if self.valid_history:
            #     axes[1].plot(self.valid_history, marker='o')
            #     axes[1].text(0.95, 0.95, f'MSE = {self.valid_history[-1]:.2f}', transform=axes[1].transAxes, va='top', ha='right')
            # axes[1].set_ylabel("MSE")
            # axes[1].set_xlabel("Iteration")

            # Fixed effect metrics
            # axes[2].plot(self.sigma_evol, label="Unexplained variance", marker='o')
            # axes[2].set_ylabel("Unexplained variance")
            # axes[2].set_xlabel("Iteration")
            # axes[2].text(0.95, 0.95, fr'$\sigma$ = {self.sigma_evol[-1]:.2f}', transform=axes[2].transAxes, va='top', ha='right')

            # Random effect metrics
            # det_re_history = [np.linalg.det(x[0]) for x in self.tau_evol]
            # trace_re_history = [np.trace(x[0]) for x in self.tau_evol]
            # axes[3].plot(det_re_history, label="|Random effect|", marker='o', zorder=2)
            # axes[3].plot(trace_re_history, label="trace(Random effect)", marker='o', zorder=1)
            # axes[3].set_ylabel("Random effect variance")
            # axes[3].set_xlabel("Iteration")
            # axes[3].text(0.95, 0.95, fr'$\mathregular{{\tau}}$ = {self.tau_evol[-1][0].item():.2f}', transform=axes[3].transAxes, va='top', ha='right')

            # Unexplained residuals distribution
            # axes[4].hist(self.eps)
            # axes[4].set_ylabel('Frequency')
            # axes[4].set_xlabel("Unexplained residual")

            # # Random effect residuals distribution
            # axes[5].hist(self.b[0].ravel())  # only first group for now
            # axes[5].set_ylabel('Frequency')
            # axes[5].set_xlabel("Random effect residual")
            for ax in axes:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True)) if ax != axes[-1] and ax != axes[-2] else None
                ax.minorticks_on()
                ax.grid(True, which='major', linewidth=0.15)
                ax.grid(True, which='minor', linestyle=':', linewidth=0.1)
            plt.show()
