from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from statsmodels.regression.mixed_linear_model import MixedLM
from tqdm import tqdm
from meml import utils
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, splu

class MEMLS:
    " Mixed effect regression model using any arbitrary fixed effect model with multiple random effects and multiple grouping factors "
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: Optional[int] = 10, gll_limit: Optional[float] = 0.001):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.gll_limit = gll_limit

        self.z: Dict[int, csr_matrix] = {}
        self.b: Dict[int, np.ndarray] = {}
        self.D: Dict[int, np.ndarray] = {}
        self.Sigma: Dict[int, np.ndarray] = {}
        self.o_k: Dict[int, int] = {}
        self.sigma2: float = None
        self.epsilon: np.ndarray = None
        self.n_obs: int = None
        self.D_history = []
        self.Sigma_history = []
        self.sigma2_history = []
        self.gll_history = []
        self.valid_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained mixed effect model.

        Random effects are valuable for understanding group differences (e.g., how much variation is due to clusters vs. fixed effects).
        They are often reported for interpretation (e.g., variance components) but not used in prediction.
        """
        self._check_inputs(X)
        return self.fe_model.predict(X)

    def fit(self, X: np.ndarray, groups: List[np.ndarray], y: np.ndarray, covariates: Optional[np.ndarray] = None,
            X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None):
        """
        Fit the mixed effect model using Expectation-Maximization algorithm
        TODO: For now it considers one random effect and one response variable
        Parameters:
        - X: 2D array (n, p) of fixed effect covariates.
        - groups: List of 1D arrays (n,) for grouping factors.
        - y: 1D array (n,) of response variable.
        - covariates: 2D array (n, q) for random effect covariates (optional).
        - X_test, y_test: Optional test data for validation.
        
        Returns:
        - Self (fitted model).
        """
        self._check_inputs(X, groups, y, covariates, X_test, y_test)
        self._initialize_em_algorithm(groups, y, covariates)
        pbar = tqdm(range(1, self.max_iter + 1), desc="MEML Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for _ in pbar:
            # Expectation step: updating fixed effects and estimating observations
            y_adj = self.adjust_y(y)
            y_pred = self.fe_model.fit(X, y_adj).predict(X)
            # Maximization step: updating residual and variance components
            self.update_parameters(y, y_pred)
            gll = self.update_gll(y, y_pred)
            self.track_variables(gll)

            pbar_desc = f"MEML Training GLL: {gll:.4f}"
            if X_test is not None:
                mse_val = self._perform_validation(X_test, y_test)
                pbar_desc += f" MSE: {mse_val:.4f}"
            pbar.set_description(pbar_desc)
            if self._is_converged(gll):
                pbar_desc = f"MEML Converged GLL: {gll:.4f}"
                pbar.set_description(pbar_desc)
                break
        return self
    
    def _check_inputs(self, x: Optional[np.ndarray] = None, groups: Optional[List[np.ndarray]] = None, y: Optional[np.ndarray] = None,
                      covariates: Optional[np.ndarray] = None, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None):
        if x is not None and (x.ndim != 2 or not np.issubdtype(x.dtype, np.floating)):
           raise ValueError("X must be 2D float")
        if groups is not None and not isinstance(groups, (list, tuple)) and any(group.ndim != 1 for group in groups):
            raise ValueError("groups must be a list or tuple of 1D arrays")
        if y is not None and (y.ndim != 1 or not np.issubdtype(y.dtype, np.floating)):
            raise ValueError("y must be 1D float")
        if covariates is not None and (covariates.ndim != 2 or not np.issubdtype(y.dtype, np.floating)):
            raise ValueError("covariates must be 2D float")
        if X_test is not None and (X_test.ndim != 2 or not np.issubdtype(X_test.dtype, np.floating)):
            raise ValueError("X_test must be 2D float")
        if y_test is not None and (y_test.ndim != 1 or not np.issubdtype(y_test.dtype, np.floating)):
            raise ValueError("y_test must be 1D float")
    
    def _initialize_em_algorithm(self, groups, y, covariates):
        """Initialize variables for the EM algorithm."""
        self.n_obs = len(y)
        # Build z_k matrices
        for k, group in enumerate(groups, 1):
            self.z[k] = utils.build_z(group, covariates)
            self.o_k[k] = len(np.unique(group))
            self.b[k] = np.zeros(self.z[k].shape[1])
            q_k = self.z[k].shape[1] // self.o_k[k]  # Random effects per level
            self.Sigma[k] = 0.1 * np.eye(q_k)  # (q_k, q_k) covariance per level
            # self.D[k] = 0.1 * np.eye(self.z[k].shape[1])
        self.epsilon = np.zeros_like(y)
        self.sigma2 = 0.1
        # f_hat = self.fe_model.fit(X, y).predict(X)
        # self.epsilon = y - f_hat
        # self.sigma2 = np.var(self.epsilon) / (len(groups) + 1)
        return self

    def adjust_y(self, y):
        """
        Adjusted response (observations - all random effects)
        """
        zb_sum = np.zeros(self.n_obs)
        for k in self.z:
            zb_sum += self.z[k] @ self.b[k]
        return y - zb_sum
    
    def update_residual_variance_mixedlm(self, y, groups, y_pred):
        """
        Update the residual and variance components using MixedLM from statsmodels
        """
        residuals = y - y_pred
        # Define mixedlm without explanatory variables
        result = MixedLM(residuals, np.ones_like(residuals), groups).fit()
        for group, effect in result.random_effects.items():
            mask = (groups == group)
            self.resid_re[mask] = effect.iloc[0]
        self.resid_unexp[...] = result.resid
        self.var_re[...] = result.cov_re
        self.var_unexp = result.scale
        return self

    def update_parameters(self, y, y_pred):
        """
        Update the residual and variance components using the EM algorithm
        """
        I_n = sparse.eye(self.n_obs, format='csc')
        V = self.sigma2 * I_n
        for k in self.z:
            # V += self.z[k] @ self.D[k] @ self.z[k].T
            D_k = sparse.kron(sparse.eye(self.o_k[k], format='csc'), self.Sigma[k])
            V += self.z[k] @ D_k @ self.z[k].T
        V = sparse.csc_matrix(V)  # Ensure V is in CSC format for efficient solving
        V_inv_y_fx = spsolve(V, y - y_pred)
        for k in self.z:
            # self.b[k] = self.D[k] @ self.z[k].T @ V_inv_y_fx
            D_k = sparse.kron(sparse.eye(self.o_k[k], format='csc'), self.Sigma[k])
            self.b[k] = D_k @ self.z[k].T @ V_inv_y_fx

        zb_sum = np.zeros(self.n_obs)
        for k in self.z:
            zb_sum += self.z[k] @ self.b[k]

        self.epsilon = y - y_pred - zb_sum

        V_inv_trace = spsolve(V, I_n).diagonal().sum()
        self.sigma2 = (1/self.n_obs) * (self.epsilon.T @ self.epsilon + self.sigma2 * (self.n_obs - self.sigma2 * V_inv_trace))

        for k in self.z:
            # D_term = self.D[k] - self.D[k] @ self.z[k].T @ spsolve(V, self.z[k] @ self.D[k])
            # self.D[k] = (1/self.o_k[k]) * (np.outer(self.b[k], self.b[k]) + D_term)

            # Update Sigma_k (q_k x q_k)
            q_k = self.z[k].shape[1] // self.o_k[k]
            bk_reshaped = self.b[k].reshape(self.o_k[k], q_k)  # (o_k, q_k)
            # Compute sum of b_ki b_ki^T
            Sigma_bb = np.zeros((q_k, q_k))
            for i in range(self.o_k[k]):
                Sigma_bb += np.outer(bk_reshaped[i], bk_reshaped[i])
            
            # Compute Z_k^T V^{-1} Z_k
            Zk_Vinv_Zk = self.z[k].T @ spsolve(V, self.z[k])  # Shape: (o_k*q_k, o_k*q_k)
            
            # Compute correction term by averaging over levels
            D_term_sum = np.zeros((q_k, q_k))
            for i in range(self.o_k[k]):
                start = i * q_k
                end = (i + 1) * q_k
                block = Zk_Vinv_Zk[start:end, start:end]  # Extract diagonal block
                D_term_sum += self.Sigma[k] - self.Sigma[k] @ block @ self.Sigma[k]
            
            # Update Sigma_k
            self.Sigma[k] = (1 / self.o_k[k]) * (Sigma_bb + D_term_sum)
        return self

    def update_gll(self, y, y_pred):
        """
        Updat the log likelihood
        """
        residuals = y - y_pred
        V = self.sigma2 * sparse.eye(self.n_obs, format='csc')
        for k in self.z:
            # V += self.z[k] @ self.D[k] @ self.z[k].T
            D_k = sparse.kron(sparse.eye(self.o_k[k], format='csc'), self.Sigma[k])
            V += self.z[k] @ D_k @ self.z[k].T
        V = sparse.csc_matrix(V)  # Ensure V is in CSC format for efficient solving
        # Using LU decomposition (more efficient for sparse matrices)
        lu = sparse.linalg.splu(V)
        # Get log determinant from the diagonal elements of U
        log_det_V = np.sum(np.log(np.abs(lu.U.diagonal())))
        
        # Solve the linear system V^-1 * residuals efficiently
        V_inv_residuals = spsolve(V, residuals)
        return -0.5 * (self.n_obs * np.log(2 * np.pi) + log_det_V + residuals.T @ V_inv_residuals)  # should we use this or complete-data log-likelihood?

    def track_variables(self, gll):
        # self.D_history.append(self.D.copy())
        self.Sigma_history.append(self.Sigma.copy())
        self.sigma2_history.append(self.sigma2)
        self.gll_history.append(gll)
        return self
    
    def _perform_validation(self, X_test, y_test):
        y_val_pred = self.predict(X_test)
        mse_val = mean_squared_error(y_test, y_val_pred)
        self.valid_history.append(mse_val)
        return mse_val

    def _is_converged(self, gll) -> bool:
        if self.gll_limit and len(self.gll_history) > 1:
            return np.abs((gll - self.gll_history[-2]) / self.gll_history[-2]) <= self.gll_limit
        return False

    def summary(self):
        """
        Plots of variance components and residuals and validation per iteration
        """
        with plt.rc_context({
            'font.family': 'Times New Roman',
            'font.size': 9,

            'lines.linewidth': 0.5,
            'lines.markersize': 1,

            'axes.titlesize': 'medium',
            'axes.linewidth': 0.2,

            'xtick.major.width': 0.2,
            'ytick.major.width': 0.2,
            'xtick.minor.width': 0.15,
            'ytick.minor.width': 0.15,

            'legend.framealpha': 1.0,
            'legend.frameon': False,

            'figure.dpi': 900,
            'figure.figsize': (3.937, 3.1496),  # 10 cm by 8 cm
            'figure.constrained_layout.use': True,

            'patch.linewidth': 0.5,}):
            fig, axes = plt.subplots(3, 2, figsize=(12/2.54, 12/2.54), layout='constrained')
            axes = axes.flatten()

            # Generalized Log-Likelihood
            axes[0].plot(self.gll_history, marker='o')
            axes[0].set_ylabel("GLL")
            axes[0].set_xlabel("Iteration")
            axes[0].text(0.95, 0.95, f'GLL = {self.gll_history[-1]:.2f}', transform=axes[0].transAxes, va='top', ha='right')

            # Validation MSE
            if self.valid_history:
                axes[1].plot(self.valid_history, marker='o')
                axes[1].text(0.95, 0.95, f'MSE = {self.valid_history[-1]:.2f}', transform=axes[1].transAxes, va='top', ha='right')
            axes[1].set_ylabel("MSE")
            axes[1].set_xlabel("Iteration")

            # Fixed effect metrics
            axes[2].plot(self.sigma2_history, label="Unexplained variance", marker='o')
            axes[2].set_ylabel("Unexplained variance")
            axes[2].set_xlabel("Iteration")
            axes[2].text(0.95, 0.95, fr'$\mathregular{{\sigma}}$ = {self.sigma2_history[-1]:.2f}', transform=axes[2].transAxes, va='top', ha='right')

            # Random effect metrics
            det_re_history = [np.linalg.det(x[1]) for x in self.Sigma_history]
            trace_re_history = [np.trace(x[1]) for x in self.Sigma_history]
            axes[3].plot(det_re_history, label="|Random effect|", marker='o', zorder=2)
            axes[3].plot(trace_re_history, label="trace(Random effect)", marker='o', zorder=1)
            axes[3].set_ylabel("Random effect variance")
            axes[3].set_xlabel("Iteration")
            axes[3].text(0.95, 0.95, fr'$\mathregular{{\tau}}$ = {self.Sigma_history[-1][1].item():.2f}', transform=axes[3].transAxes, va='top', ha='right')

            # Unexplained residuals distribution
            axes[4].hist(self.epsilon)
            axes[4].set_ylabel('Frequency')
            axes[4].set_xlabel("Unexplained residual")

            # Random effect residuals distribution
            axes[5].hist(self.b[1].ravel())  # only first group for now
            axes[5].set_ylabel('Frequency')
            axes[5].set_xlabel("Random effect residual")
            for ax in axes:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True)) if ax != axes[-1] and ax != axes[-2] else None
                ax.minorticks_on()
                ax.grid(True, which='major', linewidth=0.15)
                ax.grid(True, which='minor', linestyle=':', linewidth=0.1)
            plt.show()
