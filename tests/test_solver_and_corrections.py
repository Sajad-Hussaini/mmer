import numpy as np

from mmer.core.corrections import VarianceCorrection
from mmer.core.solver import build_solver
from mmer.core.terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual


def _build_problem(n: int, m: int, n_levels: int, slope_cols: list[int] | None):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, 3))
    groups = rng.integers(0, n_levels, size=(n, 1))

    re_term = RandomEffectTerm(group_id=0, covariates_id=slope_cols, m=m)
    re_cov = np.eye(m * re_term.q) * 1.5
    re_cov += 0.05 * np.ones_like(re_cov)
    re_term.set_cov(re_cov)

    resid_term = ResidualTerm(m=m)
    resid_term.set_cov(np.eye(m) * 0.8 + 0.1)

    realized_effect = RealizedRandomEffect(re_term, X, groups)
    realized_residual = RealizedResidual(resid_term, n)
    return X, groups, realized_effect, realized_residual


def test_woodbury_2d_solve_matches_columnwise():
    n = 8
    m = 2
    X, groups, realized_effect, realized_residual = _build_problem(n, m, n_levels=3, slope_cols=[0])
    solver = build_solver((realized_effect,), realized_residual, preconditioner=False)

    rhs = np.random.default_rng(1).normal(size=(m * n, 4))
    batched = solver.solve(rhs)
    columnwise = np.column_stack([solver.solve(rhs[:, i]) for i in range(rhs.shape[1])])

    assert batched.shape == rhs.shape
    assert np.allclose(batched, columnwise)


def test_iterative_2d_solve_matches_columnwise():
    n = 8
    m = 2
    X, groups, realized_effect, realized_residual = _build_problem(n, m, n_levels=8, slope_cols=[0])
    solver = build_solver((realized_effect,), realized_residual, preconditioner=False, cg_maxiter=200)

    rhs = np.random.default_rng(2).normal(size=(m * n, 3))
    batched = solver.solve(rhs)
    columnwise = np.column_stack([solver.solve(rhs[:, i]) for i in range(rhs.shape[1])])

    assert batched.shape == rhs.shape
    assert np.allclose(batched, columnwise)


def test_deterministic_correction_runs_and_shapes():
    n = 8
    m = 2
    _, _, realized_effect, realized_residual = _build_problem(n, m, n_levels=3, slope_cols=[0])
    solver = build_solver((realized_effect,), realized_residual, preconditioner=False)
    correction = VarianceCorrection(method='de', n_jobs=1)

    T, W = correction.compute_correction(0, solver, n_probes=4)

    assert T.shape == (m, m)
    assert W.shape == (m * realized_effect.q, m * realized_effect.q)
    assert np.isfinite(T).all()
    assert np.isfinite(W).all()