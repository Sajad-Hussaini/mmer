from typing import Optional, Dict
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class MEML:
    """
    Mixed Effect Regression Model
    This class implements a Mixed Effect Regression Model using the Expectation-Maximization (EM) algorithm. 
    It supports any fixed effect model, multiple random effects, and multiple grouping factors.
    Z: Random effect design matrix
    b: Random effect coefficients
    D: Random effect covariance matrix
    tau: Within level random effect covariance matrix 
    sigma: unexplained residuals (errors) variance
    """
    def __init__(self, fixed_effects_model: RegressorMixin, max_iter: Optional[int] = 10, gll_tol: Optional[float] = 0.001):
        self.fe_model = fixed_effects_model
        self.max_iter = max_iter
        self.gll_tol = gll_tol

        self.Z: Dict[int, sparse.csr_matrix] = {}
        self.b: Dict[int, np.ndarray] = {}
        self.tau: Dict[int, np.ndarray] = {}
        self.o: Dict[int, int] = {}
        self.q: Dict[int, int] = {}
        self.sigma: float = None
        self.eps: np.ndarray = None
        self.n: int = None

        self.I_n: sparse.csc_matrix = None
        self.zb_sum: np.ndarray = None

        self.tau_history = []
        self.sigma_history = []
        self.gll_history = []
        self.valid_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        " Predict using trained fixed effect term of the mixed effect model."
        self._check_inputs(X)
        return self.fe_model.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, random_slope_covariates: Optional[np.ndarray] = None,
            X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None):
        """
        Fit the mixed effect model using Expectation-Maximization algorithm.
        Parameters:
        - X: 2D array (n, p) of fixed effect covariates.
        - y: 1D array (n,) of response variable.
        - groups: List of 1D arrays (n,) for grouping factors.
        - random_slope_covariates: 2D array (n, q) for random effect random slope covariates (optional).
        - X_test, y_test: Optional test data for validation.
        
        Returns:
        - Self (fitted model).
        """
        self._check_inputs(X, y, groups, random_slope_covariates, X_test, y_test)
        self._initialize_EM(groups, random_slope_covariates)
        pbar = tqdm(range(1, self.max_iter + 1), desc="MEML Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}")
        for _ in pbar:
            y_adj = self.adjust_y(y)
            fX = self.fe_model.fit(X, y_adj).predict(X)
            self.update_parameters(y, fX)
            gll = self.update_gll(y, fX)
            self.track_variables(gll)

            pbar_desc = f"MEML Training GLL: {gll:.4f}"
            if X_test is not None:
                mse_val = self._perform_validation(X_test, y_test)
                pbar_desc += f" MSE: {mse_val:.4f}"
            pbar.set_description(pbar_desc)
            if self._is_converged(gll):
                pbar_desc = f"MEML Converged GLL: {gll:.4f}, Elapsed: {pbar.format_dict['elapsed']:.1f}s"
                pbar.set_description(pbar_desc)
                break
        return self
    
    def _check_inputs(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None,
                      random_slope_covariates: Optional[np.ndarray] = None, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None):
        " Validate input dimensions and types. "
        if X is not None and (X.ndim != 2 or not np.issubdtype(X.dtype, np.floating)):
           raise ValueError("X must be 2D float")
        if groups is not None and groups.ndim != 2:
            raise ValueError("groups must be a 2D objects")
        if y is not None and (y.ndim != 1 or not np.issubdtype(y.dtype, np.floating)):
            raise ValueError("y must be 1D float")
        if random_slope_covariates is not None and (random_slope_covariates.ndim != 2 or not np.issubdtype(y.dtype, np.floating)):
            raise ValueError("random_slope_covariates must be 2D float")
        if X_test is not None and (X_test.ndim != 2 or not np.issubdtype(X_test.dtype, np.floating)):
            raise ValueError("X_test must be 2D float")
        if y_test is not None and (y_test.ndim != 1 or not np.issubdtype(y_test.dtype, np.floating)):
            raise ValueError("y_test must be 1D float")
    
    def _initialize_EM(self, groups, random_slope_covariates):
        """Initialize variables for the EM algorithm."""
        self.n = groups.shape[0]
        for k in range(groups.shape[1]):
            self.Z[k], self.o[k], self.q[k] = self.design_Z(groups[:, k], random_slope_covariates)
            self.b[k] = np.zeros(self.o[k] * self.q[k])
            self.tau[k] = 0.1 * np.eye(self.q[k])
        self.eps = np.zeros(self.n)
        self.sigma = 0.1
        self.zb_sum = np.zeros(self.n)
        self.I_n = sparse.eye(self.n, format='csc')
        return self

    def adjust_y(self, y):
        " Adjusted response (observations - all random effects) "
        self.zb_sum.fill(0.0)
        for k in self.Z:
            self.zb_sum += self.Z[k] @ self.b[k]
        return y - self.zb_sum

    def update_parameters(self, y, fX):
        " Update the residual and variance components using the EM algorithm. "
        V = self.sigma * self.I_n
        for k in self.Z:
            D_k = sparse.kron(sparse.eye(self.o[k]), self.tau[k], format='csr')
            V += self.Z[k] @ D_k @ self.Z[k].T
        lu = sparse.linalg.splu(V)
        res_map = lu.solve(y - fX)
        # TODO should we fit fe_model again?
        self.zb_sum.fill(0.0)
        for k in self.Z:
            re_contribution = (self.Z[k].T @ res_map).reshape(self.o[k], self.q[k])
            self.b[k] = (re_contribution @ self.tau[k]).ravel()
            self.zb_sum += self.Z[k] @ self.b[k]

        np.subtract(y, fX, out=self.eps)
        np.subtract(self.eps, self.zb_sum, out=self.eps)

        V_inv_trace = lu.solve(self.I_n.toarray()).diagonal().sum()
        self.sigma = (self.eps.T @ self.eps + self.sigma * (self.n - self.sigma * V_inv_trace)) / self.n

        for k in self.Z:
            fisher_term_re = self.Z[k].T @ lu.solve(self.Z[k].toarray())
            fisher_term_re = fisher_term_re.reshape(self.o[k], self.q[k], self.o[k], self.q[k])
            diag_blocks_sum = np.einsum('ijil->jl', fisher_term_re)
            tau_correction = self.o[k] * self.tau[k] - self.tau[k] @ diag_blocks_sum @ self.tau[k]
            bk_reshaped = self.b[k].reshape(self.o[k], self.q[k])
            tau_bb = bk_reshaped.T @ bk_reshaped
            self.tau[k] = (tau_bb + tau_correction) / self.o[k]
        return self

    def update_gll(self, y, fX):
        """
        Updat the log likelihood
        """
        residuals = y - fX
        V = self.sigma * self.I_n
        for k in self.Z:
            D_k = sparse.kron(sparse.eye(self.o[k]), self.tau[k], format='csr')
            V += self.Z[k] @ D_k @ self.Z[k].T
        lu = sparse.linalg.splu(V)
        # Get log determinant from the diagonal elements of U
        log_det_V = np.sum(np.log(np.abs(lu.U.diagonal())))
        res_map = lu.solve(residuals)
        return -(self.n * np.log(2 * np.pi) + log_det_V + residuals.T @ res_map) / 2

    def track_variables(self, gll):
        self.tau_history.append(self.tau.copy())
        self.sigma_history.append(self.sigma)
        self.gll_history.append(gll)
        return self
    
    def _perform_validation(self, X_test, y_test):
        y_val_pred = self.predict(X_test)
        mse_val = mean_squared_error(y_test, y_val_pred)
        self.valid_history.append(mse_val)
        return mse_val

    def _is_converged(self, gll) -> bool:
        if self.gll_tol and len(self.gll_history) > 1:
            return np.abs((gll - self.gll_history[-2]) / self.gll_history[-2]) <= self.gll_tol
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
        
        nnz = self.n * q  # Number of non-zero elements
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
            axes[2].plot(self.sigma_history, label="Unexplained variance", marker='o')
            axes[2].set_ylabel("Unexplained variance")
            axes[2].set_xlabel("Iteration")
            axes[2].text(0.95, 0.95, fr'$\mathregular{{\sigma}}$ = {self.sigma_history[-1]:.2f}', transform=axes[2].transAxes, va='top', ha='right')

            # Random effect metrics
            det_re_history = [np.linalg.det(x[0]) for x in self.tau_history]
            trace_re_history = [np.trace(x[0]) for x in self.tau_history]
            axes[3].plot(det_re_history, label="|Random effect|", marker='o', zorder=2)
            axes[3].plot(trace_re_history, label="trace(Random effect)", marker='o', zorder=1)
            axes[3].set_ylabel("Random effect variance")
            axes[3].set_xlabel("Iteration")
            axes[3].text(0.95, 0.95, fr'$\mathregular{{\tau}}$ = {self.tau_history[-1][0].item():.2f}', transform=axes[3].transAxes, va='top', ha='right')

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
