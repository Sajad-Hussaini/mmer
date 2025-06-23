import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.base import RegressorMixin, clone
from tqdm import tqdm
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
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihood = []
        self._is_converged = False

    def prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_indices: dict):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        self.num_obs, self.num_responses = y.shape
        self.num_groups = groups.shape[1]
        self.slope_indices = random_slope_indices

        Z_matrices, self.num_random_effects, self.num_levels = utils.random_effect_design_matrices(X, groups, self.slope_indices)
        Z_crossprods = utils.crossprod_design_matrices(Z_matrices)

        self.residuals_covariance = np.eye(self.num_responses)
        self.random_effects_covariance = {k: np.eye(self.num_responses * q_k) for k, q_k in self.num_random_effects.items()}
        
        self.fe_models = [clone(self.fe_model) for _ in range(self.num_responses)] if not isinstance(self.fe_model, list) else self.fe_model
        if len(self.fe_models) != self.num_responses:
            raise ValueError(f"Expected {self.num_responses} fixed effects models, got {len(self.fe_models)}")
        
        marginal_residuals = self.compute_marginal_residuals(X, y, 0.0)
        return marginal_residuals, Z_matrices, Z_crossprods
    
    def compute_marginal_residuals(self, X, y, effects_sum):
        """
        Compute marginal residuals by fitting the fixed effects models to the adjusted response variables.
        """
        y_adj = y - effects_sum
        for m in range(self.num_responses):
            y_adj[:, m] = self.fe_models[m].fit(X, y_adj[:, m]).predict(X)
        return y - y_adj
    
    def _V_matvec(self, x_vec, Z_matrices):
        """
        Computes the matrix-vector product V @ x_vec, where V = Σ(I_M ⊗ Z_k) D_k (I_M ⊗ Z_k)^T + R is the marginal covariance.
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        n, M = self.num_obs, self.num_responses
        x_mat = x_vec.reshape((M, n)).T
        Vx = x_mat @ self.residuals_covariance

        for k in range(self.num_groups):
            q_k, o_k = self.num_random_effects[k], self.num_levels[k]
            Z_k = Z_matrices[k]
            A_k = self._Wk_T_matvec(x_vec, Z_matrices, k)
            A_k = A_k.reshape((M, q_k * o_k)).T
            Vx += Z_k @ A_k
        Vx.T.ravel()
        return Vx

    def _Wk_matvec(self, x_vec, Z_matrices, k):
        """
        Computes the matrix-vector product W_k @ x_vec, where W_k = (I_M ⊗ Z_k) D_k maps a vector from
        the random effects space (pre-weighted by D_k) to the observation space.
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        M = self.num_responses
        Z_k = Z_matrices[k]
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        tau_k = self.random_effects_covariance[k]

        x_mat = x_vec.reshape((M * q_k, o_k)).T
        A_k = x_mat @ tau_k
        A_k = A_k.reshape((o_k, M, q_k)).transpose(1, 2, 0).reshape((M, q_k * o_k)).T
        B_k = Z_k @ A_k
        B_k = B_k.T.ravel()  # (M*n, )
        return B_k

    def _Wk_T_matvec(self, x_vec, Z_matrices, k):
        """
        Computes the matrix-vector product W_k^T @ x_vec, where W_k^T = D_k (I_M ⊗ Z_k)^T maps a vector from
        the observation space back to the random effects space (post-weighted by D_k).
        It leverages the Kronecker structure to avoid full matrix construction.
        """
        n, M = self.num_obs, self.num_responses
        Z_k = Z_matrices[k]
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        tau_k = self.random_effects_covariance[k]

        x_mat = x_vec.reshape((M, n)).T
        A_k = Z_k.T @ x_mat
        A_k = A_k.reshape((q_k, o_k, M)).transpose(1, 2, 0).reshape((o_k, M * q_k))
        B_k = A_k @ tau_k
        B_k = B_k.reshape((o_k, M, q_k)).transpose(1, 2, 0).ravel()  # (M*q_k*o_k, )
        return B_k

    def compute_residuals_covariance(self, V_op, eps, Z_matrices, Z_crossprods):
        """
        Compute the residual covariance matrix.
        Uses symmetry of the covariance matrix to reduce computations.
        """
        S = eps.T @ eps
        T = np.zeros((self.num_responses, self.num_responses))
        
        # Only compute upper triangular part (including diagonal)
        for m1 in range(self.num_responses):
            for m2 in range(m1, self.num_responses):  # Using symmetry: only m2 >= m1
                trace_sum = 0.0
                for k in range(self.num_groups):
                    sigma_k_block = self.compute_sigma_k_res_block(V_op, Z_matrices, k, m1, m2)
                    trace_sum += (Z_crossprods[k] @ sigma_k_block).trace()
                T[m1, m2] = trace_sum
                T[m2, m1] = trace_sum
        self.residuals_covariance = (S + T) / self.num_obs + 1e-6 * np.eye(self.num_responses)

    def compute_sigma_k_res_block(self, V_op, Z_matrices, k, m1, m2):
        """
        Compute Sigma_k = D_k - D_k (I_M ⊗ Z_k)^T V^{-1} (I_M ⊗ Z_k) D_k for response block (m1, m2).
        """
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        M = self.num_responses

        tau_k_block = self.random_effects_covariance[k][m1 * q_k : (m1 + 1) * q_k, m2 * q_k : (m2 + 1) * q_k]
        D_k_m1_m2 = sparse.kron(tau_k_block, sparse.eye_array(o_k, format='csr'), format='csr')
        sigma_k_m1_m2 = np.zeros((q_k * o_k, q_k * o_k))
        base_mat = np.eye(M * q_k * o_k)[m2 * q_k * o_k : (m2 + 1) * q_k * o_k]
        col = 0
        for ej in base_mat:
            wj = self._Wk_matvec(ej, Z_matrices, k)  # ej basis vector extracting col j of wk
            xj, _ = cg(V_op, wj)
            wk_xj = self._Wk_T_matvec(xj, Z_matrices, k)
            sigma_k_m1_m2[:, col] = wk_xj[m1 * q_k * o_k : (m1 + 1) * q_k * o_k]
            col += 1

        sigma_k_m1_m2 = D_k_m1_m2 - sigma_k_m1_m2
        return sigma_k_m1_m2

    def compute_random_effects_covariance(self, mu, V_op, Z_matrices):
        """
        Compute the random effects covariance matrix.
        """
        for k in range(self.num_groups):
            o_k, q_k = self.num_levels[k], self.num_random_effects[k]
            mu_k = mu[k]
            sum_tau = np.zeros((self.num_responses * q_k, self.num_responses * q_k))
            # Compute indices for all levels
            m_idx = np.arange(self.num_responses)[:, None]
            q_idx = np.arange(q_k)[None, :]
            base = m_idx * q_k * o_k + q_idx * o_k
            for j in range(o_k):
                indices = (base + j).ravel()
                mu_k_j = mu_k[indices]
                sigma_k_block = self.compute_sigma_k_level_block(V_op, Z_matrices, k, j)
                sum_tau += np.outer(mu_k_j, mu_k_j) + sigma_k_block
            self.random_effects_covariance[k] = sum_tau / o_k + 1e-6 * np.eye(self.num_responses * q_k)
    
    def compute_sigma_k_level_block(self, V_op, Z_matrices, k, l1):
        """
        Compute Sigma_k = D_k - D_k (I_M ⊗ Z_k)^T V^{-1} (I_M ⊗ Z_k) D_k for level block (l1, l1).
        """
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        M = self.num_responses
        
        # Compute indices for all levels
        m_idx = np.arange(M)[:, None]
        q_idx = np.arange(q_k)[None, :]
        base = m_idx * q_k * o_k + q_idx * o_k
        indices = (base + l1).ravel()

        D_k_l1 = self.random_effects_covariance[k]

        sigma_k_l1 = np.zeros((M * q_k, M * q_k))
        base_mat = np.eye(M * q_k * o_k)[indices]
        col = 0
        for ej in base_mat:
            wj = self._Wk_matvec(ej, Z_matrices, k)
            xj, _ = cg(V_op, wj)
            wk_xj = self._Wk_T_matvec(xj, Z_matrices, k)
            sigma_k_l1[:, col] = wk_xj[indices]
            col += 1

        sigma_k_l1 = D_k_l1 - sigma_k_l1
        return sigma_k_l1
    
    def compute_log_likelihood(self, marginal_residuals, V_inv_eps, V_op):
        """
        Compute the log-likelihood of the marginal distribution of the residuals (the marginal log-likelihood)
        """
        log_det_V = utils.slq_logdet(V_op, self.num_responses * self.num_obs)
        log_likelihood = -(self.num_responses * self.num_obs * np.log(2 * np.pi) + log_det_V + marginal_residuals.T @ V_inv_eps) / 2
        return log_likelihood
    
    def compute_marginal_covariance(self, num_obs, n_level, Z_matrices):
        """
        Compute the marginal covariance matrix V and the random effects covariance matrices D.
        """
        V = sparse.kron(self.residuals_covariance, sparse.eye_array(num_obs, format='csr'), format='csr')
        for k in range(self.num_groups):
            D_k = sparse.kron(self.random_effects_covariance[k], sparse.eye_array(n_level[k], format='csr'), format='csr')
            Z_block_k = utils.block_diag_design_matrix(self.num_responses, Z_matrices[k])
            V += Z_block_k @ D_k @ Z_block_k.T
        return V

    def compute_mu(self, V_inv_eps, Z_matrices):
        """
        Compute the conditional mean of the random effects by leveraging the kronecker structure.
        """
        mu = {}
        for k in range(self.num_groups):
            mu[k] = self._Wk_T_matvec(V_inv_eps, Z_matrices, k)
        return mu
    
    def sum_random_effects(self, mu, Z_matrices):
        """
        Compute the sum of random effects contributions for all groups.
        """
        n, M = self.num_obs, self.num_responses
        effects_sum = np.zeros((n, M))
        for k, mu_k in mu.items():
            q_k, o_k = self.num_random_effects[k], self.num_levels[k]
            Z_k = Z_matrices[k]
            A_k = mu_k.reshape((M, q_k * o_k)).T
            effects_sum += Z_k @ A_k
        return effects_sum

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_indices: dict = None):
        """
        Fit the multivariate mixed effects model using EM algorithm.

        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
            y: (n_samples, M) array of M response variables.
            groups: (n_samples, K) array of K grouping factors.
            random_slope_indices: List of (n_samples, q_k) arrays for random slopes per group (optional).

        Returns:
            Self (fitted model).
        """
        marginal_residuals, Z_matrices, Z_crossprods = self.prepare_data(X, y, groups, random_slope_indices)
        size = self.num_responses * self.num_obs
        V_op = LinearOperator(shape=(size, size), matvec=lambda x_vec: self._V_matvec(x_vec, Z_matrices))
        pbar = tqdm(range(1, self.max_iter + 1), desc="Fitting model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for iter_ in pbar:
            rhs = marginal_residuals.T.ravel()
            V_inv_eps, _ = cg(V_op, rhs)
            mu = self.compute_mu(V_inv_eps, Z_matrices)
            if iter_ > 2 and abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2]) < self.tol:
                pbar.set_description("Model Converged")
                self._is_converged = True
                break
            effects_sum = self.sum_random_effects(mu, Z_matrices)
            marginal_residuals = self.compute_marginal_residuals(X, y, effects_sum)
            eps = marginal_residuals - effects_sum
            self.compute_residuals_covariance(V_op, eps, Z_matrices, Z_crossprods)
            self.compute_random_effects_covariance(mu, V_op, Z_matrices)
            log_likelihood = self.compute_log_likelihood(rhs, V_inv_eps, V_op)
            self.log_likelihood.append(log_likelihood)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses using the fitted fixed effects models.
        
        Parameters:
            X: (n_samples, n_features) array of fixed effect covariates.
        
        Returns:
            (n_samples, M) array of predicted responses.
        """
        if not self.fe_models:
            raise ValueError("Model must be fitted before prediction.")
        n = X.shape[0]
        fX = np.zeros((n, self.num_responses))
        for m in range(self.num_responses):
            fX[:, m] = self.fe_models[m].predict(X)
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
        groups = np.zeros((n, self.num_groups), dtype=int)
        for i in range(n):
            Z_matrices, _, n_level = utils.random_effect_design_matrices(X[i:i+1], groups[i:i+1], self.slope_indices)
            Z_blocks = utils.block_diag_design_matrices(Z_matrices, self.num_responses)
            V_i, _ = self.compute_marginal_covariance(1, n_level, Z_blocks)
            y_sampled[i] = np.random.multivariate_normal(fX[i], V_i)
        return y_sampled
    
    def compute_random_effects_and_residuals(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        Compute residuals (num_obs x num_obs) and random effects (num_responses x num_random_effects x num_levels).
        """
        self.num_obs, _ = y.shape
        Z_matrices, _, n_level = utils.random_effect_design_matrices(X, groups, self.slope_indices)
        Z_blocks = utils.block_diag_design_matrices(Z_matrices, self.num_responses)
        splu, D = self.splu_decomposition(self.num_obs, n_level, Z_blocks)

        marginal_residuals = y - self.predict(X)
        V_inv_eps = splu.solve(marginal_residuals.ravel(order='F'))
        mu = self.compute_mu(V_inv_eps, D, Z_blocks)

        effects_sum = self.sum_random_effects(mu, Z_matrices)
        eps = marginal_residuals - effects_sum
        return mu, eps

    def summary(self):
        """
        Display a summary of the fitted multivariate mixed effects model.
        """
        if not self.fe_models:
            raise ValueError("Model must be fitted before calling summary.")

        # Print summary statistics
        indent0 = ""
        indent1 = "   "
        indent2 = "       "

        print("\n" + indent0 + "Multivariate Mixed Effects Model Summary")
        print("=" * 50)
        print(indent1 + f"FE Model: {type(self.fe_models[0]).__name__}")
        print(indent1 + f"Iterations: {len(self.log_likelihood)}")
        print(indent1 + f"Converged: {self._is_converged}")
        print(indent1 + f"Log-Likelihood: {self.log_likelihood[-1]:.2f}")
        print(indent1 + f"No. Observations: {self.num_obs}")
        print(indent1 + f"No. Response Variables: {self.num_responses}")
        print(indent1 + f"No. Grouping Variables: {self.num_groups}")
        print("-" * 50)
        print(indent1 + f"Residual (Unexplained) Variances")
        print(indent2 + "{:<10} {:>10}".format("Response", "Variance"))
        for m in range(self.num_responses):
            print(indent2 + "{:<10} {:>10.4f}".format(m + 1, self.residuals_covariance[m, m]))
        print("-" * 50)
        print(indent1 + f"Random Effects Variances")
        print(indent2 + "{:<8} {:<10} {:<15} {:>10}".format("Group", "Response", "Random Effect", "Variance"))
        for k in range(self.num_groups):
            for i in range(self.num_responses):
                for j in range(self.num_random_effects[k]):
                    idx = i * self.num_random_effects[k] + j
                    effect_name = "Intercept" if j == 0 else f"Slope {j}"
                    var = self.random_effects_covariance[k][idx, idx]
                    print(indent2 + "{:<8} {:<10} {:<15} {:>10.4f}".format(k + 1, i + 1, effect_name, var))
        print("\n")