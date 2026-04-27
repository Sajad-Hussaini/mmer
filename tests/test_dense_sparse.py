import numpy as np
from scipy import sparse
import time

m, n, qo = 100, 10000, 200
dense = np.random.rand(m, n)
Z = sparse.random(n, qo, density=0.01, format='csr')

t0 = time.time()
for _ in range(10):
    dense @ Z
print("dense @ Z (csr):", time.time() - t0)

t0 = time.time()
for _ in range(10):
    (Z.T @ dense.T).T
print("(Z.T @ dense.T).T:", time.time() - t0)

Z_csc = Z.tocsc()
t0 = time.time()
for _ in range(10):
    dense @ Z_csc
print("dense @ Z (csc):", time.time() - t0)
