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
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_indices: dict):
        """
        Initialize the model parameters and design matrices based on input data.
        """
        self.num_obs, self.num_responses = y.shape
        self.num_groups = groups.shape[1]
        self.slope_indices = random_slope_indices

        Z_matrices, self.num_random_effects, self.num_levels = utils.random_effect_design_matrices(X, groups, self.slope_indices)
        # Z_blocks = utils.block_diag_design_matrices(Z_matrices, self.num_responses)
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
        Compute the marginal residuals by fitting the fixed effects models to the adjusted response variables.
        """
        y_adj = y - effects_sum
        fX = np.zeros_like(y_adj)
        for m in range(self.num_responses):
            fX[:, m] = self.fe_models[m].fit(X, y_adj[:, m]).predict(X)
        marginal_residuals = y - fX
        return marginal_residuals
    
    def V_matvec(self, x_vec, Z_matrices):
        """
        Efficiently computes matrix-vector multiplication without full matrix construction and leverage the Kronecker structure.
        """
        n, M = self.num_obs, self.num_responses
        x_mat = x_vec.reshape((M, n)).T
        Vx = x_mat @ self.residuals_covariance

        for k in range(self.num_groups):
            q_k, o_k = self.num_random_effects[k], self.num_levels[k]
            Z_k = Z_matrices[k]
            A_k = self.Wk_T_matvec(x_vec, Z_matrices, k)
            A_k = A_k.reshape((M, q_k * o_k)).T
            Vx += Z_k @ A_k
        return Vx.T.ravel()
    
    def V_matmat(self, X_matrix, Z_matrices):
        n, M = self.num_obs, self.num_responses
        num_cols = X_matrix.shape[1] # P

        # 1. x_mat = x_vec.reshape((M, n)).T
        X_mat_reshaped = X_matrix.reshape((M, n, num_cols)).transpose(1, 0, 2) # (n, M, P)
        
        # 2. Vx = x_mat @ self.residuals_covariance
        # X_mat_reshaped is (n, M, P)
        # self.residuals_covariance is (M, M)
        Vx_matrix_reshaped = np.einsum('nmr,rs->nsr', X_mat_reshaped, self.residuals_covariance) # 'n' for n, 'm' for M, 'r' for P, 's' for M
                                                                                                # Result: (n, M, P)

        for k in range(self.num_groups):
            q_k, o_k = self.num_random_effects[k], self.num_levels[k]
            Z_k = Z_matrices[k] # (n, q_k * o_k)
            
            # 3. A_k = self.Wk_T_matmat(X_matrix, Z_matrices, k)
            # Wk_T_matmat returns (M*q_k*o_k, P)
            A_k_full = self.Wk_T_matmat(X_matrix, Z_matrices, k) # (M * q_k * o_k, P)

            # 4. A_k = A_k.reshape((M, q_k * o_k)).T
            # A_k_full is (M * q_k * o_k, P)
            A_k_reshaped_for_Z = A_k_full.reshape((M, q_k * o_k, num_cols)).transpose(1, 0, 2)
            # Result: (q_k * o_k, M, P)

            # 5. Vx += Z_k @ A_k
            # Vx_matrix_reshaped is (n, M, P)
            # Z_k is (n, q_k * o_k)
            # A_k_reshaped_for_Z is (q_k * o_k, M, P)
            Vx_matrix_reshaped += np.einsum('zq,qmp->zmp', Z_k, A_k_reshaped_for_Z) # 'z' for n, 'q' for q_k*o_k, 'm' for M, 'p' for P
                                                                                # Result: (n, M, P)

        # 6. return Vx.T.ravel() -> (M*n, P)
        return Vx_matrix_reshaped.transpose(1, 0, 2).reshape((M * n, num_cols))
    
    def e_step(self, marginal_residuals, Z_matrices):
        """
        Perform the E-step of the EM algorithm to compute the conditional expectation and covariance of the random effects.
        """
        n, M = self.num_obs, self.num_responses
        size = n * M
        V_op = LinearOperator(shape=(size, size), matvec=lambda x_vec: self.V_matvec(x_vec, Z_matrices))
        rhs = marginal_residuals.T.ravel()
        V_inv_eps, _ = cg(V_op, rhs)
        # log_likelihood = self.compute_log_likelihood(rhs, V_inv_eps, V_op)
        log_likelihood = -1.0
        mu = self.compute_mu(V_inv_eps, Z_matrices)
        # sigma = self.compute_sigma(V_op, Z_matrices)
        return mu, V_op, log_likelihood
    
    # def e_step(self, marginal_residuals, Z_matrices, Z_blocks):
    #     """
    #     Perform the E-step of the EM algorithm to compute the conditional expectation and covariance of the random effects.
    #     """
    #     splu, D = self.splu_decomposition(self.num_obs, self.num_levels, Z_blocks)
    #     m_eps_stack = marginal_residuals.T.ravel()
    #     V_inv_eps = splu.solve(m_eps_stack)
    #     log_likelihood = self.compute_log_likelihood(m_eps_stack, V_inv_eps, splu)
    #     mu = self.compute_mu(V_inv_eps, Z_matrices)
    #     sigma = self.compute_sigma(D, splu, Z_blocks)
    #     return mu, sigma, log_likelihood
    
    def m_step(self, X, y, mu, V_op, Z_matrices, Z_crossprods):
        """
        Perform the E-step of the EM algorithm to update the fixed effects functions, residual, and random effects covariance matrices.
        """
        effects_sum = self.sum_random_effects(mu, Z_matrices)
        marginal_residuals = self.compute_marginal_residuals(X, y, effects_sum)
        eps = marginal_residuals - effects_sum
        self.compute_residuals_covariance(V_op, eps, Z_matrices, Z_crossprods)
        self.compute_random_effects_covariance(mu, V_op, Z_matrices)
        return marginal_residuals
    
    # def compute_residuals_covariance(self, sigma, eps, Z_crossprods):
    #     """
    #     the residual covariance matrix residuals_covariance.
    #     """
    #     S = eps.T @ eps
    #     T = np.zeros((self.num_responses, self.num_responses))
    #     for m1 in range(self.num_responses):
    #         for m2 in range(self.num_responses):
    #             trace_sum = 0.0
    #             for k in range(self.num_groups):
    #                 o_k, q_k = self.num_levels[k], self.num_random_effects[k]
    #                 idx1 = slice(m1 * q_k * o_k, (m1 + 1) * q_k * o_k)
    #                 idx2 = slice(m2 * q_k * o_k, (m2 + 1) * q_k * o_k)
    #                 sigma_k_block = sigma[k][idx1, idx2]
    #                 trace_sum += (Z_crossprods[k] @ sigma_k_block).trace()
    #             T[m1, m2] = trace_sum
    #     self.residuals_covariance = (S + T) / self.num_obs + 1e-6 * np.eye(self.num_responses)

    def compute_residuals_covariance(self, V_op, eps, Z_matrices, Z_crossprods):
        """
        the residual covariance matrix residuals_covariance.
        """
        S = eps.T @ eps
        T = np.zeros((self.num_responses, self.num_responses))
        for m1 in range(self.num_responses):
            for m2 in range(self.num_responses):
                trace_sum = 0.0
                for k in range(self.num_groups):
                    # o_k, q_k = self.num_levels[k], self.num_random_effects[k]
                    # idx1 = slice(m1 * q_k * o_k, (m1 + 1) * q_k * o_k)
                    # idx2 = slice(m2 * q_k * o_k, (m2 + 1) * q_k * o_k)
                    sigma_k_block = self.compute_sigma_k_response_ij(V_op, Z_matrices, k, m1, m2)
                    # sigma_k_block = sigma[k][idx1, idx2]
                    trace_sum += (Z_crossprods[k] @ sigma_k_block).trace()
                T[m1, m2] = trace_sum
        self.residuals_covariance = (S + T) / self.num_obs + 1e-6 * np.eye(self.num_responses)

    def compute_random_effects_covariance(self, mu, V_op, Z_matrices):
        """
        Update the random effects covariance matrix random_effects_covariance.
        """
        for k in range(self.num_groups):
            o_k, q_k = self.num_levels[k], self.num_random_effects[k]
            mu_k = mu[k]
            # sigma_k = sigma[k]
            sum_tau = np.zeros((self.num_responses * q_k, self.num_responses * q_k))
            for j in range(o_k):
                indices = []  # Indices for level j across all responses and effect types
                for m in range(self.num_responses):
                    for q in range(q_k):
                        idx = m * q_k * o_k + q * o_k + j
                        indices.append(idx)
                mu_k_j = mu_k[indices]
                sigma_k_block = self.compute_sigma_k_level_i(V_op, Z_matrices, k, j)
                # sigma_k_block = sigma_k[np.ix_(indices, indices)]
                sum_tau += np.outer(mu_k_j, mu_k_j) + sigma_k_block
            self.random_effects_covariance[k] = sum_tau / o_k + 1e-6 * np.eye(self.num_responses * q_k)

    # def compute_random_effects_covariance(self, mu, sigma):
    #     """
    #     Update the random effects covariance matrix random_effects_covariance.
    #     """
    #     for k in range(self.num_groups):
    #         o_k, q_k = self.num_levels[k], self.num_random_effects[k]
    #         mu_k = mu[k]
    #         sigma_k = sigma[k]
    #         sum_tau = np.zeros((self.num_responses * q_k, self.num_responses * q_k))
    #         for j in range(o_k):
    #             indices = []  # Indices for level j across all responses and effect types
    #             for m in range(self.num_responses):
    #                 for q in range(q_k):
    #                     idx = m * q_k * o_k + q * o_k + j
    #                     indices.append(idx)
    #             mu_k_j = mu_k[indices]
    #             sigma_k_block = sigma_k[np.ix_(indices, indices)]
    #             sum_tau += np.outer(mu_k_j, mu_k_j) + sigma_k_block
    #         self.random_effects_covariance[k] = sum_tau / o_k + 1e-6 * np.eye(self.num_responses * q_k)

    # def compute_log_likelihood(self, marginal_residuals, V_inv_eps, splu):
    #     """
    #     Compute the log-likelihood of the marginal distribution of the residuals (the marginal log-likelihood)
    #     """
    #     log_det_V = np.sum(np.log(np.abs(splu.U.diagonal())))
    #     log_likelihood = -(self.num_responses * self.num_obs * np.log(2 * np.pi) + log_det_V + marginal_residuals.T @ V_inv_eps) / 2
    #     return log_likelihood
    
    def slq_logdet(self, V_op, dim, num_probes=10, m=20):
        """
        Approximate log(det(V)) using Stochastic Lanczos Quadrature (SLQ).
        V_op: LinearOperator for V
        dim: dimension of V
        num_probes: number of random vectors
        m: number of Lanczos steps
        """
        logdet_est = 0.0
        for _ in range(num_probes):
            v = np.random.choice([-1, 1], size=dim)
            v = v / np.linalg.norm(v)
            Q = np.zeros((dim, m+1))
            alpha = np.zeros(m)
            beta = np.zeros(m)
            Q[:, 0] = v
            for j in range(m):
                w = V_op @ Q[:, j]
                if j > 0:
                    w -= beta[j-1] * Q[:, j-1]
                alpha[j] = np.dot(Q[:, j], w)
                w -= alpha[j] * Q[:, j]
                beta[j] = np.linalg.norm(w)
                if beta[j] < 1e-10 or j == m-1:
                    break
                Q[:, j+1] = w / beta[j]
            T = np.diag(alpha[:j+1]) + np.diag(beta[:j], 1) + np.diag(beta[:j], -1)
            eigvals = np.linalg.eigvalsh(T)
            logdet_est += np.sum(np.log(eigvals))
        return dim * logdet_est / (num_probes * m)
    
    def compute_log_likelihood(self, marginal_residuals, V_inv_eps, V_op):
        """
        Compute the log-likelihood of the marginal distribution of the residuals (the marginal log-likelihood)
        """
        log_det_V = self.slq_logdet(V_op, int(self.num_responses * self.num_obs))
        log_likelihood = -(self.num_responses * self.num_obs * np.log(2 * np.pi) + log_det_V + marginal_residuals.T @ V_inv_eps) / 2
        return log_likelihood
    
    def compute_marginal_covariance(self, num_obs, n_level, Z_blocks):
        """
        Compute the marginal covariance matrix V and the random effects covariance matrices D.
        """
        D = {k: sparse.kron(tau_k, sparse.eye_array(n_level[k], format='csr'), format='csr') for k, tau_k in self.random_effects_covariance.items()}
        V = sparse.kron(self.residuals_covariance, sparse.eye_array(num_obs, format='csr'), format='csr')
        for k, D_k in D.items():
            V += Z_blocks[k] @ D_k @ Z_blocks[k].T
        return V, D

    def splu_decomposition(self, num_obs, n_level, Z_blocks):
        """
        Compute the sparse LU decomposition of the marginal covariance matrix V.
        """
        V, D = self.compute_marginal_covariance(num_obs, n_level, Z_blocks)
        return sparse.linalg.splu(V.tocsc()), D

    @staticmethod
    def compute_muo(V_inv_eps, D, Z_blocks):
        """
        Compute the conditional mean of the random effects.
        """
        return {k: D_k @ Z_blocks[k].T @ V_inv_eps for k, D_k in D.items()}
    

    def compute_mu(self, V_inv_eps, Z_matrices):
        """
        Compute the conditional mean of the random effects efficiently using kronecker structure.
        """
        mu = {}
        for k in range(self.num_groups):
            mu[k] = self.Wk_T_matvec(V_inv_eps, Z_matrices, k)
        return mu

    def Wk_matvec(self, x_vec, Z_matrices, k):
        """
        Efficiently computes matrix-vector multiplication without full matrix construction and leverage the Kronecker structure.
        It applies (I_M ⊗ Z_k)D_k to a vector x_vec.
        """
        n, M = self.num_obs, self.num_responses
        Z_k = Z_matrices[k]
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        tau_k = self.random_effects_covariance[k]

        x_mat = x_vec.reshape((M * q_k, o_k)).T
        A_k = x_mat @ tau_k
        A_k = A_k.reshape((o_k, M, q_k)).transpose(1, 2, 0).reshape((M, q_k * o_k)).T
        B_k = Z_k @ A_k
        B_k.T.ravel()  # (M*n, )
        return B_k
    
    def Wk_matmat(self, X_matrix, Z_matrices, k):
        n, M = self.num_obs, self.num_responses
        num_cols = X_matrix.shape[1] # P

        Z_k = Z_matrices[k] # (n, q_k * o_k)
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        tau_k = self.random_effects_covariance[k] # (M * q_k, M * q_k)

        # 1. x_mat = x_vec.reshape((M * q_k, o_k)).T
        # X_matrix is (M*q_k*o_k, P)
        X_mat_reshaped_step1 = X_matrix.reshape((M * q_k, o_k, num_cols)).transpose(1, 0, 2)
        # Result: (o_k, M * q_k, P)

        # 2. A_k = x_mat @ tau_k
        # X_mat_reshaped_step1 is (o_k, M * q_k, P)
        # tau_k is (M * q_k, M * q_k)
        # This requires an einsum for batch matrix multiplication
        A_k_step2 = np.einsum('oqc,qd->odc', X_mat_reshaped_step1, tau_k) # 'oqc' for (o_k, M*q_k, P), 'qd' for (M*q_k, M*q_k)
                                                                    # Indices: o=o_k, q=M*q_k, c=P, d=M*q_k
                                                                    # Result: (o_k, M*q_k, P)
        
        # 3. A_k = A_k.reshape((o_k, M, q_k)).transpose(1, 2, 0).reshape((M, q_k * o_k)).T
        A_k_step3_1 = A_k_step2.reshape((o_k, M, q_k, num_cols)) # (o_k, M, q_k, P)
        A_k_step3_2 = A_k_step3_1.transpose(1, 2, 0, 3) # (M, q_k, o_k, P)
        A_k_step3_3 = A_k_step3_2.reshape((M, q_k * o_k, num_cols)) # (M, q_k * o_k, P)
        A_k_final_for_Z = A_k_step3_3.transpose(1, 0, 2) # (q_k * o_k, M, P)

        # 4. B_k = Z_k @ A_k
        # Z_k is (n, q_k * o_k)
        # A_k_final_for_Z is (q_k * o_k, M, P)
        B_k_matrix_reshaped = np.einsum('zq,qmp->zmp', Z_k, A_k_final_for_Z) # 'z' for n, 'q' for q_k*o_k, 'm' for M, 'p' for P
                                                                    # Result: (n, M, P)

        # 5. B_k.T.ravel() -> (M*n, P)
        return B_k_matrix_reshaped.transpose(1, 0, 2).reshape((M * n, num_cols))

    def Wk_T_matvec(self, x_vec, Z_matrices, k):
        """
        Efficiently computes matrix-vector multiplication without full matrix construction and leverage the Kronecker structure.
        It aplies D_k(I_M ⊗ Z_k)^T to a vector x_vec.
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
    
    def Wk_T_matmat(self, X_matrix, Z_matrices, k):
        n, M = self.num_obs, self.num_responses
        num_cols = X_matrix.shape[1] # P

        Z_k = Z_matrices[k] # (n, q_k * o_k)
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        tau_k = self.random_effects_covariance[k] # (M * q_k, M * q_k)

        # 1. x_mat = x_vec.reshape((M, n)).T
        X_mat_reshaped_step1 = X_matrix.reshape((M, n, num_cols)).transpose(1, 0, 2)
        # Result: (n, M, P)

        # 2. A_k = Z_k.T @ x_mat
        # Z_k.T is (q_k * o_k, n)
        # X_mat_reshaped_step1 is (n, M, P)
        A_k_step2 = np.einsum('qn,nmr->qmr', Z_k.T, X_mat_reshaped_step1) # 'q' for q_k*o_k, 'n' for n, 'm' for M, 'r' for P
                                                                    # Result: (q_k * o_k, M, P)

        # 3. A_k = A_k.reshape((q_k, o_k, M)).transpose(1, 2, 0).reshape((o_k, M * q_k))
        A_k_step3_1 = A_k_step2.reshape((q_k, o_k, M, num_cols)) # (q_k, o_k, M, P)
        A_k_step3_2 = A_k_step3_1.transpose(1, 2, 0, 3) # (o_k, M, q_k, P)
        A_k_step3_3 = A_k_step3_2.reshape((o_k, M * q_k, num_cols)) # (o_k, M * q_k, P)

        # 4. B_k = A_k @ tau_k
        # A_k_step3_3 is (o_k, M * q_k, P)
        # tau_k is (M * q_k, M * q_k)
        # This needs an einsum for batch matrix multiplication
        B_k_matrix_reshaped = np.einsum('oqc,qd->odc', A_k_step3_3, tau_k) # 'o' for o_k, 'q' for M*q_k, 'c' for P, 'd' for M*q_k
                                                                    # Result: (o_k, M * q_k, P)

        # 5. B_k = B_k.reshape((o_k, M, q_k)).transpose(1, 2, 0).ravel() -> (M*q_k*o_k, P)
        return B_k_matrix_reshaped.reshape((o_k, M, q_k, num_cols)).transpose(1, 2, 0, 3).reshape((M * q_k * o_k, num_cols))

    # @staticmethod
    # def compute_sigma(D, splu, Z_blocks):
    #     """
    #     Compute the conditional covariance of the random effects.
    #     """
    #     sigma = {}
    #     for k, D_k in D.items():
    #         Im_Z_k = Z_blocks[k]
    #         Im_Z_D = Im_Z_k @ D_k
    #         V_inv_Im_Z_D = sparse.csr_array(splu.solve(Im_Z_D.toarray()))
    #         sigma[k] = D_k - D_k @ Im_Z_k.T @ V_inv_Im_Z_D
    #     return sigma
    
    def compute_sigma(self, V_op, Z_matrices):
        """
        Compute the conditional covariance of the random effects.
        """
        sigma = {}
        for k in range(self.num_groups):
            sigma[k] = self.compute_sigma_k(V_op, Z_matrices, k)
        return sigma
    
    def compute_sigma_k(self, V_op, Z_matrices, k):
        """
        Compute Sigma_k = D_k - D_k (I_M ⊗ Z_k)^T V^{-1} (I_M ⊗ Z_k) D_k
        using matrix-free CG.
        """
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        M, n = self.num_responses, self.num_obs
        tau_k = self.random_effects_covariance[k]  # (M*qk x M*qk)
        D_k = sparse.kron(tau_k, sparse.eye_array(o_k, format='csr'), format='csr')

        # Loop over standard basis vectors or low-rank approximation
        sigma_k = np.zeros((M * q_k * o_k, M * q_k * o_k))

        for j in range(M * q_k * o_k):
            ej = np.zeros(M * q_k * o_k)
            ej[j] = 1.0

            wj = self.Wk_matvec(ej, Z_matrices, k)  # its like extracting col j of wk 
            xj, _ = cg(V_op, wj)
            
            wk_xj = self.Wk_T_matvec(xj, Z_matrices, k)
            sigma_k[:, j] = wk_xj

        sigma_k = D_k - sigma_k
        return sigma_k
    
    def compute_sigma_k_response_ij(self, V_op, Z_matrices, k, m1, m2):
        """
        Compute Sigma_k = D_k - D_k (I_M ⊗ Z_k)^T V^{-1} (I_M ⊗ Z_k) D_k for response m1 and m2
        using matrix-free CG.
        """
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        M, n = self.num_responses, self.num_obs
        tau_k = self.random_effects_covariance[k]  # (M*qk x M*qk)
        tau_k_block = tau_k[m1 * q_k : (m1 + 1) * q_k, m2 * q_k : (m2 + 1) * q_k]
        D_k_m1_m2 = sparse.kron(tau_k_block, sparse.eye_array(o_k, format='csr'), format='csr')

        # Loop over standard basis vectors or low-rank approximation
        sigma_k_m1_m2 = np.zeros((q_k * o_k, q_k * o_k))
        col = 0
        for j in range(m2 * q_k * o_k, (m2 + 1) * q_k * o_k):
            ej = np.zeros(M * q_k * o_k)
            ej[j] = 1.0

            wj = self.Wk_matvec(ej, Z_matrices, k)  # its like extracting col j of wk 
            xj, _ = cg(V_op, wj)
            wk_xj = self.Wk_T_matvec(xj, Z_matrices, k)
            sigma_k_m1_m2[:, col] = wk_xj[m1 * q_k * o_k : (m1 + 1) * q_k * o_k]
            col += 1

        sigma_k_m1_m2 = D_k_m1_m2 - sigma_k_m1_m2
        return sigma_k_m1_m2
    
    def compute_sigma_k_level_i(self, V_op, Z_matrices, k, l1):
        """
        Compute Sigma_k = D_k - D_k (I_M ⊗ Z_k)^T V^{-1} (I_M ⊗ Z_k) D_k for levels l1 and l2 across all responses and effects
        using matrix-free CG.
        """
        q_k, o_k = self.num_random_effects[k], self.num_levels[k]
        M, n = self.num_responses, self.num_obs
        tau_k = self.random_effects_covariance[k]  # (M*qk x M*qk)

        
        indices = []  # Indices for level j across all responses and effect types
        for m in range(M):
            for q in range(q_k):
                idx = m * q_k * o_k + q * o_k + l1
                indices.append(idx)

        D_k_l1_l1 = tau_k   # for a level i it is actually tau_k itself?

        # Loop over standard basis vectors or low-rank approximation
        sigma_k_l1_l1 = np.zeros((M * q_k, M * q_k))
        col = 0
        for j in indices:
            ej = np.zeros(M * q_k * o_k)
            ej[j] = 1.0

            wj = self.Wk_matvec(ej, Z_matrices, k)  # its like extracting col j of wk 
            xj, _ = cg(V_op, wj)
            wk_xj = self.Wk_T_matvec(xj, Z_matrices, k)
            sigma_k_l1_l1[:, col] = wk_xj[indices]
            col += 1

        sigma_k_l1_l1 = D_k_l1_l1 - sigma_k_l1_l1
        return sigma_k_l1_l1

    def sum_random_effectso(self, mu, Z_blocks, num_obs):
        """
        Compute the sum of random effects contributions for all groups.
        """
        effects_sum = np.zeros((self.num_obs, self.num_responses))
        for k, mu_k in mu.items():
            effects_sum += (Z_blocks[k] @ mu_k).reshape((self.num_obs, self.num_responses), order='F')
        return effects_sum
    
    def sum_random_effects(self, mu, Z_matrices):
        """
        Compute the sum of random effects contributions for all groups.
        """
        n, M = self.num_obs, self.num_responses
        effects_sum = np.zeros((n, M))
        for k, mu_k in mu.items():
            q_k, o_k = self.num_random_effects[k], self.num_levels[k]
            Z_k = Z_matrices[k]
            A_mat = mu_k.reshape((q_k * o_k, M), order='F')
            effects_sum += Z_k @ A_mat
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
        pbar = tqdm(range(1, self.max_iter + 1), desc="Preparing data", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        marginal_residuals, Z_matrices, Z_crossprods = self._prepare_data(X, y, groups, random_slope_indices)
        pbar.set_description("Fitting model")
        for iter_ in pbar:
            pbar.set_postfix_str("E-step")
            mu, V_op, log_likelihood = self.e_step(marginal_residuals, Z_matrices)
            self.log_likelihood.append(log_likelihood)
            # if iter_ > 2 and abs((self.log_likelihood[-1] - self.log_likelihood[-2]) / self.log_likelihood[-2]) < self.tol:
            #     pbar.set_description("Model Converged")
            #     self._is_converged = True
            #     break
            pbar.set_postfix_str("M-step")
            marginal_residuals = self.m_step(X, y, mu, V_op, Z_matrices, Z_crossprods)
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