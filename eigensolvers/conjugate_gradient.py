import numpy as np

#
# Alg
#


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> np.ndarray:
    x0 = np.random.rand(A.shape[0])
    r0 = b - A @ x0
    p0 = r0.copy()

    for k in range(max_iter):
        # Cache intermediate
        A_p0 = np.einsum("ij,j->i", A, p0)

        # Step length
        α_n = np.dot(r0, r0) / np.dot(p0, A_p0)

        # Approximate solution
        x_n = x0 + α_n * p0

        # Residual
        r_n = r0 - α_n * A_p0  # for linear equations
        # print(f"Check r_n^t r0   = 0  {np.dot(r_n, r0):.2e}")
        print(f"Residual norm:        {np.linalg.norm(r_n):.2e}")

        # Improvement this step
        β_n = np.dot(r_n, r_n) / np.dot(r0, r0)

        # New search direction
        p_n = r_n + β_n * p0
        # print(f"Check p_n^t A p0 = 0  {np.dot(p_n, A_p0):.2e}")

        # Update
        x0 = x_n.copy()
        r0 = r_n.copy()
        p0 = p_n.copy()

    return x_n


#
# Main
#

np.random.seed(18)
N = 5000
H = np.random.rand(N, N)
H = H + H.T
H *= 1e-3
b = np.random.rand(N)

S_t, U_t = np.linalg.eigh(H)

# Make it positive semi-definite
H_psd = np.einsum(
    "ij,jk,kl->il", U_t, np.diag(np.random.rand(N) + 1), U_t.T, optimize=True
)

np.set_printoptions(linewidth=160)
x_n = conjugate_gradient(H_psd, b, max_iter=25)
x_exact = np.linalg.solve(H_psd, b)
# print(x_n - x_exact)
print(f"Condition number: {np.linalg.cond(H_psd):.2e}")
print(
    f"Relative error:   {np.linalg.norm(x_n - x_exact) / np.linalg.norm(x_exact):.2e}"
)
# print(S_t[:10] - S_t[0])
