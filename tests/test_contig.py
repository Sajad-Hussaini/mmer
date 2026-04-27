import numpy as np

m = 100
q = 1
o = 200
K = 60

Dx = np.random.rand(m * q * o, K)
xr = Dx.reshape((m, q * o, K))
print("xr[0] is C-contiguous:", xr[0].flags['C_CONTIGUOUS'])
print("xr[0] is F-contiguous:", xr[0].flags['F_CONTIGUOUS'])
