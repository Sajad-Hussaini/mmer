import numpy as np
from scipy import sparse
import sys
import os

sys.path.insert(0, os.path.abspath('src'))

from mmer.core.terms import RandomEffectTerm, ResidualTerm, RealizedRandomEffect, RealizedResidual
from mmer.core.operator import VLinearOperator

np.random.seed(0)
n = 100
m = 2
o1 = 5
o2 = 10

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

v_op = VLinearOperator((re1, re2), rr)
x = np.random.rand(m * n)

y1 = v_op @ x
y2 = v_op @ x

print("Difference between two runs:", np.linalg.norm(y1 - y2))

V_dense = np.kron(rr.term.cov, np.eye(n))
for r in [re1, re2]:
    Z_k = np.kron(np.eye(m), r.Z.toarray())
    D_k = np.kron(r.term.cov, np.eye(r.o))
    V_dense += Z_k @ D_k @ Z_k.T
    
y_exact = V_dense @ x
print("Exact diff:", np.linalg.norm(y1 - y_exact))
