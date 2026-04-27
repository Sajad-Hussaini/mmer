import numpy as np
from scipy import sparse
import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath('src'))

from mmer.core.terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual
from mmer.core.solver import build_solver

np.random.seed(0)
n = 10000
m = 100
o1 = 200
o2 = 400

groups = np.zeros((n, 2), dtype=int)
groups[:, 0] = np.random.randint(0, o1, n)
groups[:, 1] = np.random.randint(0, o2, n)

X = np.ones((n, 1))

term1 = RandomEffectTerm(0, None, m)
term2 = RandomEffectTerm(1, None, m)
term1.set_cov(np.eye(m))
term2.set_cov(np.eye(m))

resid_term = ResidualTerm(m)
resid_term.set_cov(np.eye(m) + 0.1 * np.random.rand(m, m))

re1 = RealizedRandomEffect(term1, X, groups)
re2 = RealizedRandomEffect(term2, X, groups)
rr = RealizedResidual(resid_term, n)

print("Building solver...")
try:
    solver = build_solver((re1, re2), rr)
    print("Solver built:", type(solver))
except Exception as e:
    print("Error:", e)
