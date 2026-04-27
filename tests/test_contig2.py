import numpy as np

m, q, o, K = 100, 1, 200, 60
x_vec = np.random.rand(m * q * o, K)
xr = x_vec.reshape((m * q, o, K))
xr_flat = xr.reshape((m * q, o * K))

print("x_vec contiguous:", x_vec.flags['C_CONTIGUOUS'])
print("xr_flat contiguous:", xr_flat.flags['C_CONTIGUOUS'])
