import numpy as np
from scipy import sparse
from mmer.core.terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual
from mmer.core.solver import SolverContext
from mmer.core.operator import VLinearOperator

def test_woodbury():
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

if __name__ == "__main__":
    test_woodbury()
