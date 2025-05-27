from typing import Optional, Dict
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import copy
from .style import style

class MERM:
    """
    Mixed Effect Regression Model
    This class implements a Mixed Effect Regression Model using the Expectation-Maximization (EM) algorithm. 
    It supports any fixed effect model, multiple random effects, and multiple grouping factors.
    Main Parameters baed on the Paper:
        Z: Random effect design matrix
        b: Random effect coefficients (interpcept and slopes)
        D: Full random effect covariance matrix
        tau: Within-level (shared) random effect covariance matrix 
        sigma: unexplained residuals (errors) variance
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: Optional[int] = 10, mll_tol: Optional[float] = 0.001):
        """
        Initialize the MERM model with a fixed effects model and EM algorithm parameters.
        Parameters:
            fixed_effects_model: A scikit-learn regressor model for fixed effects (e.g., LinearRegression, MLPRegressor).
            max_iter: Maximum number of iterations for the EM algorithm (default: 10).
            mll_tol: Tolerance for convergence based on the marginal log-likelihood (default: 0.001).
        """
        self.fem = fixed_effects_model
        self.max_iter = max_iter
        self.mll_tol = mll_tol

        # Z: Random effect design matrices for each grouping factor (sparse matrices).
        self.Z: Dict[int, sparse.csr_matrix] = {}
        # b: Random effect coefficients for each grouping factor.
        self.b: Dict[int, np.ndarray] = {}
        # tau: Covariance matrices per level for random effects of each grouping factor.
        self.tau: Dict[int, np.ndarray] = {}
        # o: Number of levels (unique groups) for each grouping factor.
        self.o: Dict[int, int] = {}
        # q: Number of random effects (intercept + slopes) for each grouping factor.
        self.q: Dict[int, int] = {}
        self.D: Dict[int, int] = {}
        # sigma: Variance of unexplained residuals (errors).
        self.sigma: float = None
        # eps: Residuals after accounting for fixed and random effects.
        self.eps: np.ndarray = None
        # n: Total number of observations.
        self.n: int = None

        self.I_n: sparse.csc_matrix = None
        self.zb_sum: np.ndarray = None
        self.tau_evol = []
        self.sigma_evol = []
        self.mll_evol = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        " Predict using trained fixed effect term of the mixed effect model."
        self._validate_Xy(X)
        return self.fem.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_covariates: Optional[np.ndarray] = None):
        """
        Fit the mixed effect model using Expectation-Maximization algorithm.
        Parameters:
            X: 2D array (n, p) of p fixed effect covariates.
            y: 1D array (n,) of response variable.
            groups: 2D array (n, m) of m grouping factors.
            random_slope_covariates: 2D array (n, q) for random slope covariates (optional).
        Returns:
            Self (fitted model).
        """
        self._validate_Xy(X, y)._validate_groups(groups, random_slope_covariates)
        self._initialize_EM(groups, random_slope_covariates)
        pbar = tqdm(range(1, self.max_iter + 1), desc="MERM Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for _ in pbar:
            y_adj = self._adjust_y(y)
            fX = self.fem.fit(X, y_adj).predict(X)
            self._MLE_based_EM(y, fX)
            mll = self._update_mll()
            self._track_parameters(mll)
            if self._is_converged():
                pbar_desc = f"MERM Converged"
                pbar.set_description(pbar_desc)
                break
        return self
    
    def performance(self, X_test: np.ndarray, y_test: np.ndarray):
        self._validate_Xy(X_test, y_test)
        y_test_pred = self.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        return mse_test

    def _validate_Xy(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        " Validate input X and y dimensions and types. "
        if X.ndim != 2 or not np.issubdtype(X.dtype, np.floating):
            raise ValueError("X must be 2D float")
        if y is not None and (y.ndim != 1 or not np.issubdtype(y.dtype, np.floating)):
            raise ValueError("y must be 1D float")
        return self
    
    def _validate_groups(self, groups: np.ndarray, random_slope_covariates: Optional[np.ndarray] = None):
        " Validate input groups and random slope covariates dimensions and types. "
        if groups.ndim != 2:
            raise ValueError("groups must be a 2D objects")
        if random_slope_covariates is not None and (random_slope_covariates.ndim != 2 or not np.issubdtype(random_slope_covariates.dtype, np.floating)):
            raise ValueError("random_slope_covariates must be 2D float")
        return self
    
    def _initialize_EM(self, groups, random_slope_covariates):
        """Initialize variables for the EM algorithm."""
        self.n = groups.shape[0]
        for k in range(groups.shape[1]):
            self.Z[k], self.o[k], self.q[k] = self.design_Z(groups[:, k], random_slope_covariates)
            self.b[k] = np.zeros(self.o[k] * self.q[k])
            self.tau[k] = 0.1 * np.eye(self.q[k])
        self.eps = np.zeros(self.n)
        self.eps_marginal = np.zeros(self.n)
        self.sigma = 0.1
        self.zb_sum = np.zeros(self.n)
        self.I_n = sparse.eye(self.n, format='csc')
        return self

    def _adjust_y(self, y):
        " Compute the adjusted response. "
        return y - self.zb_sum
    
    def _lu_decompose_V(self):
        " LU decomposition of the covariance matrix (V) of the observed data. "
        V = self.sigma * self.I_n
        for k in self.Z:
            V = self._update_V_per_group(V, k)
        return sparse.linalg.splu(V)
    
    def _update_V_per_group(self, V, k):
        " Update covariance matrix (V) of the observed data per group. "
        # D_k = sparse.kron(sparse.eye(self.o[k]), self.tau[k], format='csr')
        self.D[k] = sparse.kron(sparse.eye(self.o[k]), self.tau[k], format='csr')
        V += self.Z[k] @ self.D[k] @ self.Z[k].T
        return V
    
    def _update_bk_eps(self, lu, y, fX):
        " Update values of the random effects and errors. "
        np.subtract(y, fX, out=self.eps_marginal)
        scaled_res = lu.solve(self.eps_marginal)
        self.zb_sum.fill(0.0)
        for k in self.Z:
            self._update_b_zb_per_group(scaled_res, k)
        np.subtract(self.eps_marginal, self.zb_sum, out=self.eps)
        return self
    
    def _update_b_zb_per_group(self, scaled_res, k):
        " Update values of the random effects per group. "
        re_contribution = (self.Z[k].T @ scaled_res).reshape(self.o[k], self.q[k])
        self.b[k][:] = (re_contribution @ self.tau[k]).ravel()
        self.zb_sum += self.Z[k] @ self.b[k]
        return self
    
    def _update_cond_cov_per_group(self, lu, k):
        " Update the conditional covariance matrix of the random effects. "
        cond_k = self.D[k] - self.D[k] @ self.Z[k].T @ lu.solve(self.Z[k] @ self.D[k])
        return cond_k
    
    # def _update_sigma(self, lu):
    #     " Update the unexplained residuals (errors) variance. "
    #     V_inv_trace = lu.solve(self.I_n.toarray()).diagonal().sum()
    #     self.sigma = (self.eps.T @ self.eps + self.sigma * (self.n - self.sigma * V_inv_trace)) / self.n
    #     return self
    def _update_sigma(self, lu, cond_cov):
        " Update the unexplained residuals (errors) variance. "
        self.sigma = self.eps.T @ self.eps
        for k in self.Z:
            self.sigma += np.trace(cond_cov[k] @ (self.Z[k].T @ self.Z[k]).toarray())
        self.sigma /= self.n
        return self
    
    def _update_tau(self, lu):
        " Update the random effects shared covariance matrices. "
        for k in self.Z:
            self._update_tau_per_group(lu, k)
        return self
    
    def _update_tau_per_group(self, lu, k):
        " Update the random effects shared covariance matrices per group. "
        fisher_term_re = self.Z[k].T @ lu.solve(self.Z[k].toarray())
        fisher_term_re = fisher_term_re.reshape(self.o[k], self.q[k], self.o[k], self.q[k])
        sum_diag_blocks = np.einsum('ijil->jl', fisher_term_re)
        tau_correction = self.o[k] * self.tau[k] - self.tau[k] @ sum_diag_blocks @ self.tau[k]
        bk_reshaped = self.b[k].reshape(self.o[k], self.q[k])
        tau_bb = bk_reshaped.T @ bk_reshaped
        self.tau[k][:] = (tau_bb + tau_correction) / self.o[k]
        return self

    def _MLE_based_EM(self, y, fX):
        " Maximum likelihood estimation (MLE)-based EM for the random effects, errors, and variances. "
        # 1. Expectation step: update the random effects and errors
        lu = self._lu_decompose_V()
        self._update_bk_eps(lu, y, fX)
        # 2. Maximization step: update the unexplained residuals and variances
        self._update_sigma(lu)
        self._update_tau(lu)
        return self

    def _update_mll(self):
        """
        Updat the marginal log likelihood
        """
        lu = self._lu_decompose_V()
        # Compute the log determinant using LU decomposition U diagonal elements
        log_det_V = np.sum(np.log(np.abs(lu.U.diagonal())))
        scaled_res = lu.solve(self.eps_marginal)
        mll = -(self.n * np.log(2 * np.pi) + log_det_V + self.eps_marginal.T @ scaled_res) / 2
        return mll

    def _track_parameters(self, mll):
        self.tau_evol.append(copy.deepcopy(self.tau))
        self.sigma_evol.append(self.sigma)
        self.mll_evol.append(mll)
        return self

    def _is_converged(self) -> bool:
        if self.mll_tol and len(self.mll_evol) > 1:
            return np.isclose(self.mll_evol[-1], self.mll_evol[-2], rtol=self.mll_tol)
        return False
    
    def design_Z(self, group: np.ndarray, random_slope_covariates: np.ndarray = None):
        """
        Random effect design matrix for a grouping factor.
        
        Parameters:
        - group: 1D array of shape (n,) with grouping factor levels (e.g., Earthquake IDs).
        - random_slope_covariates: 2D array of shape (n, q) for random effect random slope covariates (None for intercept only).
        
        Returns:
        - Sparse matrix of shape (n, o * q), where o is number of levels, q is number of random effects.
        """
        levels = np.unique(group)
        o = len(levels)
        q = 1 if random_slope_covariates is None else 1 + random_slope_covariates.shape[1]
        # Map levels to 0-based indices
        level_map = {level: idx for idx, level in enumerate(levels)}
        level_indices = np.array([level_map[level] for level in group])
        # Number of non-zero elements
        nnz = self.n * q
        rows = np.zeros(nnz, dtype=int)
        cols = np.zeros(nnz, dtype=int)
        data = np.zeros(nnz, dtype=float)
        for i in range(self.n):
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
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n, o * q)), o, q
    
    def random_effects_for_grouping(self, k: int = 0):
        " Get individual random effects for a specific grouping factor k. "
        b_reshaped = self.b[k].reshape(self.o[k], self.q[k])
        level_indices = (self.Z[k][:, ::self.q[k]].toarray() == 1).argmax(axis=1)
        return b_reshaped[level_indices]

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
            axes[2].plot(self.sigma_evol, label="Unexplained variance", marker='o')
            axes[2].set_ylabel("Unexplained variance")
            axes[2].set_xlabel("Iteration")
            axes[2].text(0.95, 0.95, fr'$\sigma$ = {self.sigma_evol[-1]:.2f}', transform=axes[2].transAxes, va='top', ha='right')

            # Random effect metrics
            det_re_history = [np.linalg.det(x[0]) for x in self.tau_evol]
            trace_re_history = [np.trace(x[0]) for x in self.tau_evol]
            axes[3].plot(det_re_history, label="|Random effect|", marker='o', zorder=2)
            axes[3].plot(trace_re_history, label="trace(Random effect)", marker='o', zorder=1)
            axes[3].set_ylabel("Random effect variance")
            axes[3].set_xlabel("Iteration")
            axes[3].text(0.95, 0.95, fr'$\mathregular{{\tau}}$ = {self.tau_evol[-1][0].item():.2f}', transform=axes[3].transAxes, va='top', ha='right')

            # Unexplained residuals distribution
            axes[4].hist(self.eps)
            axes[4].set_ylabel('Frequency')
            axes[4].set_xlabel("Unexplained residual")

            # Random effect residuals distribution
            axes[5].hist(self.b[0].ravel())  # only first group for now
            axes[5].set_ylabel('Frequency')
            axes[5].set_xlabel("Random effect residual")
            for ax in axes:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True)) if ax != axes[-1] and ax != axes[-2] else None
                ax.minorticks_on()
                ax.grid(True, which='major', linewidth=0.15)
                ax.grid(True, which='minor', linestyle=':', linewidth=0.1)
            plt.show()
