import numpy as np

np.random.seed(18)

A = np.random.rand(3, 3)
print("A\n" + "#" * 80 + "\n", A)

Q, R = np.linalg.qr(A)

print("Q\n" + "#" * 80 + "\n", Q)
print("R\n" + "#" * 80 + "\n", R)
