import numpy as np
from scipy import sparse
import time

np.random.seed(0)
m = 100
n = 10000
q = 1
o = 200
K = 60

Z = sparse.random(n, q*o, density=0.01, format='csr')
x_vec = np.random.rand(m * n * K)

# Original kronZ_T
def orig():
    xr = x_vec.reshape((m, n, K))
    out_orig = np.empty((m, q * o, K))
    for i in range(m):
        out_orig[i] = Z.T @ xr[i]
    return out_orig.reshape(m * q * o, K)

# Optimized kronZ_T
def opt():
    xr_swapped = x_vec.reshape((m, n, K)).swapaxes(0, 1).reshape((n, m * K))
    Y = Z.T @ xr_swapped
    out_opt = Y.reshape((q * o, m, K)).swapaxes(0, 1)
    return out_opt.reshape(m * q * o, K)

t0 = time.time()
for _ in range(10): orig()
print("Original:", time.time() - t0)

t0 = time.time()
for _ in range(10): opt()
print("Optimized:", time.time() - t0)
