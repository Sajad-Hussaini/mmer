from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def cg_solve(
    operator: LinearOperator,
    rhs: np.ndarray,
    preconditioner: LinearOperator | None = None,
    x0: np.ndarray | None = None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> np.ndarray:
    """
    Solve a symmetric positive definite linear system with CG.

    The helper centralizes the tolerance and warning behavior so all CG-based
    paths use the same policy.
    """
    solution, info = cg(
        A=operator,
        b=rhs,
        M=preconditioner,
        x0=x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
    )
    if info != 0:
        print(f"Warning: CG info={info}")
    return solution