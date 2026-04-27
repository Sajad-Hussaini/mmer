import numpy as np
from mmer.core.corrections import VarianceCorrection
from mmer.core.solver import build_solver
from mmer.core.terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual

def _build_problem(n: int, m: int, n_levels1: int, n_levels2: int):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, 2))
    
    # Two grouping factors
    groups1 = rng.integers(0, n_levels1, size=(n, 1))
    groups2 = rng.integers(0, n_levels2, size=(n, 1))

    # RE 1: intercept + 1 slope => q=2
    re_term1 = RandomEffectTerm(group_id=0, covariates_id=[0], m=m)
    re_cov1 = np.eye(m * re_term1.q) * 1.5 + 0.5
    re_term1.set_cov(re_cov1)
    
    # RE 2: intercept only => q=1
    re_term2 = RandomEffectTerm(group_id=1, covariates_id=None, m=m)
    re_cov2 = np.eye(m * re_term2.q) * 1.0 + 0.2
    re_term2.set_cov(re_cov2)

    resid_term = ResidualTerm(m=m)
    resid_term.set_cov(np.eye(m) * 0.8 + 0.1)

    groups = np.hstack([groups1, groups2])
    re1 = RealizedRandomEffect(re_term1, X, groups)
    re2 = RealizedRandomEffect(re_term2, X, groups)
    rr = RealizedResidual(resid_term, n)
    
    return X, (groups1, groups2), (re1, re2), rr

# Setup
n = 50
m = 3
X, groups, realized_effects, realized_residual = _build_problem(n, m, n_levels1=5, n_levels2=3)

# Build the solver
solver = build_solver(realized_effects, realized_residual)

# We want to test k=0, which has q=2 (intercept + slope)
k = 0

print(f"Testing Grouping Factor {k} (q={realized_effects[k].q})")

# 1. Deterministic Estimation (Ground Truth)
from mmer.core.corrections import compute_cov_correction_de, compute_cov_correction_bste

T_de, W_de = compute_cov_correction_de(k, solver, n_jobs=1, backend='threading')

print("--- DE (Ground Truth) ---")
print("T_de (Residual trace corrections):\n", T_de)
print("\nW_de (Covariance correction, upper-left 4x4 block):\n", W_de[:4, :4])


# 2. Block Stochastic Trace Estimation (BSTE)
# Use a high number of probes so the Monte Carlo variance is low
T_bste, W_bste = compute_cov_correction_bste(k, solver, n_probes=2000, n_jobs=1, backend='threading')

print("\n--- BSTE (1000 probes) ---")
print("T_bste (Residual trace corrections):\n", T_bste)
print("\nW_bste (Covariance correction, upper-left 4x4 block):\n", W_bste[:4, :4])

print("\n--- Differences (BSTE - DE) ---")
print("Max absolute difference in T:", np.max(np.abs(T_bste - T_de)))
print("Max absolute difference in W:", np.max(np.abs(W_bste - W_de)))
