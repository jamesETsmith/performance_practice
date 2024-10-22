import numpy as np

np.random.seed(18)

A = np.random.rand(3, 3)
print(A)

Q, R = np.linalg.qr(A)

print(Q)
print(R)
