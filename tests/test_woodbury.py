import numpy as np
from scipy import sparse
from mmer.core.terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual
from mmer.core.solver import SolverContext
from mmer.core.operator import VLinearOperator
import time

def test_woodbury():
    print("--- Small Exact Test ---")
    # Set up synthetic data
    n = 100
    m = 2
    groups = np.random.randint(0, 5, size=(n, 1))
    X = np.random.randn(n, 2)
    
    # Set up terms
    re_term = RandomEffectTerm(group_id=0, covariates_id=[0], m=m)
    re_term.set_cov(np.array([[2.0, 0.5, 0.1, 0.0],
                               [0.5, 1.5, 0.0, 0.1],
                               [0.1, 0.0, 1.0, 0.2],
                               [0.0, 0.1, 0.2, 0.8]]))
    
    res_term = ResidualTerm(m=m)
    res_term.set_cov(np.array([[1.0, 0.3], [0.3, 0.8]]))
    
    # Set up realized terms
    re_realized = RealizedRandomEffect(re_term, X, groups)
    res_realized = RealizedResidual(res_term, n)
    
    # Solve some random rhs
    np.random.seed(42)
    rhs = np.random.randn(m * n)
    
    # Initialize SolverContext
    ctx = SolverContext((re_realized,), res_realized, preconditioner=False)
    
    # Compare with scipy.sparse.linalg.cg using VLinearOperator
    V_op = VLinearOperator((re_realized,), res_realized)
    V_mat = sparse.linalg.aslinearoperator(V_op).matmat(np.eye(m*n))
    
    x_exact = np.linalg.solve(V_mat, rhs)
    
    x_woodbury, _, _ = ctx.solve_woodbury(rhs)
    
    diff = np.linalg.norm(x_exact - x_woodbury) / (np.linalg.norm(x_exact) + 1e-15)
    print(f"Relative error: {diff:.2e}")
    if diff < 1e-8:
        print("Success! Woodbury solve matches exact inverse.")
    else:
        print("Mismatch in results.")

def test_woodbury_large():
    print("\n--- Large Dataset Test ---")
    n = 100_000
    m = 10
    groups = np.random.randint(0, 50, size=(n, 1))
    X = np.random.randn(n, 2)
    
    # Set up terms
    # For m=10 and q=2 (intercept+1 slope), cov shape is 20x20
    re_term = RandomEffectTerm(group_id=0, covariates_id=[0], m=m)
    cov_re = np.eye(20) * 2.0 + 0.1
    np.fill_diagonal(cov_re, cov_re.diagonal() + 1.0) # Ensure pos-def
    re_term.set_cov(cov_re)
    
    res_term = ResidualTerm(m=m)
    cov_res = np.eye(m) * 1.0 + 0.2
    res_term.set_cov(cov_res)
    
    re_realized = RealizedRandomEffect(re_term, X, groups)
    res_realized = RealizedResidual(res_term, n)
    
    # Initialize SolverContext
    ctx = SolverContext((re_realized,), res_realized, preconditioner=True, cg_maxiter=100)
    
    # Random RHS vector (1D) and matrix (2D blocks for STE probes)
    np.random.seed(42)
    rhs_1d = np.random.randn(m * n)
    rhs_2d = np.random.randn(m * n, 50)  # 50 simultaneous trace estimation probes
    
    t0 = time.time()
    x_woodbury_1d, _, _ = ctx.solve_woodbury(rhs_1d)
    t1 = time.time()
    print(f"Woodbury 1D Solve Time (N={n}, m={m}): {t1 - t0:.4f}s")
    
    t0 = time.time()
    x_woodbury_2d, _, _ = ctx.solve_woodbury(rhs_2d)
    t1 = time.time()
    print(f"Woodbury 2D Block Solve Time (50 probes): {t1 - t0:.4f}s")

def test_woodbury_huge():
    print("\n--- Huge Dataset Test ---")
    n = 100_000
    m = 100
    # Let's say 100 unique levels for the grouping factor
    levels = 100 
    groups = np.random.randint(0, levels, size=(n, 1))
    X = np.random.randn(n, 2)
    
    # Set up terms
    # For m=100 and q=2 (intercept+1 slope), cov shape is 200x200
    re_term = RandomEffectTerm(group_id=0, covariates_id=[0], m=m)
    cov_re = np.eye(200) * 2.0 + 0.1
    np.fill_diagonal(cov_re, cov_re.diagonal() + 1.0) # Ensure pos-def
    re_term.set_cov(cov_re)
    
    res_term = ResidualTerm(m=m)
    cov_res = np.eye(m) * 1.0 + 0.2
    np.fill_diagonal(cov_res, cov_res.diagonal() + 1.0)
    res_term.set_cov(cov_res)
    
    re_realized = RealizedRandomEffect(re_term, X, groups)
    res_realized = RealizedResidual(res_term, n)
    
    # Initialize SolverContext
    ctx = SolverContext((re_realized,), res_realized, preconditioner=True, cg_maxiter=100)
    
    # Random RHS vector (1D) and matrix (2D blocks for STE probes)
    np.random.seed(42)
    rhs_1d = np.random.randn(m * n)            # 1D vector of length 10,000,000
    rhs_2d = np.random.randn(m * n, 5)         # 2D matrix (5 simultaneous probes)
    
    t0 = time.time()
    x_woodbury_1d, _, _ = ctx.solve_woodbury(rhs_1d)
    t1 = time.time()
    print(f"Woodbury 1D Solve Time (N={n}, m={m}, DOF={m*n}): {t1 - t0:.4f}s")
    
    t0 = time.time()
    x_woodbury_2d, _, _ = ctx.solve_woodbury(rhs_2d)
    t1 = time.time()
    print(f"Woodbury 2D Block Solve Time (5 probes): {t1 - t0:.4f}s")

if __name__ == "__main__":
    test_woodbury()
    test_woodbury_large()
    test_woodbury_huge()

