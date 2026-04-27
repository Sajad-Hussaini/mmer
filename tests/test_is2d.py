import numpy as np
from scipy import sparse

np.random.seed(0)
m = 3
n = 5
q = 2
o = 4
K = 2

Z = sparse.random(n, q*o, density=0.5, format='csr')

x_vec = np.random.rand(m * n * K)

# Original kronZ_T
xr = x_vec.reshape((m, n, K))
out_orig = np.empty((m, q * o, K))
for i in range(m):
    out_orig[i] = Z.T @ xr[i]
out_orig_flat = out_orig.reshape(m * q * o, K)

# Optimized kronZ_T
xr_swapped = x_vec.reshape((m, n, K)).swapaxes(0, 1).reshape((n, m * K))
Y = Z.T @ xr_swapped
out_opt = Y.reshape((q * o, m, K)).swapaxes(0, 1)
out_opt_flat = out_opt.reshape(m * q * o, K)

print("kronZ_T exact match:", np.allclose(out_orig_flat, out_opt_flat))

# Original kronZ
x_vec2 = np.random.rand(m * q * o * K)
xr2 = x_vec2.reshape((m, q * o, K))
out2_orig = np.empty((m, n, K))
for i in range(m):
    out2_orig[i] = Z @ xr2[i]
out2_orig_flat = out2_orig.reshape(m * n, K)

# Optimized kronZ
xr2_swapped = x_vec2.reshape((m, q * o, K)).swapaxes(0, 1).reshape((q * o, m * K))
Y2 = Z @ xr2_swapped
out2_opt = Y2.reshape((n, m, K)).swapaxes(0, 1)
out2_opt_flat = out2_opt.reshape(m * n, K)

print("kronZ exact match:", np.allclose(out2_orig_flat, out2_opt_flat))
