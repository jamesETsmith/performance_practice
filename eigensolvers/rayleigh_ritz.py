import numpy as np


def rayleigh_ritz(
    A: np.ndarray, X: np.ndarray, max_iter: int = 25, tol: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    Xi = X.copy()
    li_old = 1e10

    print(f"Starting Rayleigh-Ritz with tol {tol}")
    print(f"{'Iter':<8} {'λ_max':<20} {'Δλ/λ':<16} {'Residual':<16}")
    for i in range(max_iter):
        li = np.einsum("i,ij,j->", Xi.conj(), A, Xi) / np.linalg.norm(Xi)

        # Eigenvectors for the inverse problem are the same
        y = np.linalg.solve(A - li * np.eye(A.shape[0]), Xi)
        Xi = y / np.linalg.norm(y)

        Ri = A @ Xi - li * Xi
        print(f"{i:<8} {li:<20} {abs(li - li_old):<16e} {np.linalg.norm(Ri):<16e}")
        if np.linalg.norm(Ri) < tol:
            print("Converged")
            return li, Xi

        li_old = li

    return None, None


np.random.seed(18)
N = 1000
nx = 1
H = np.random.rand(N, N)
H = H + H.T
psi = np.random.rand(N)

np.set_printoptions(linewidth=160)

li, xi = rayleigh_ritz(H, psi, max_iter=50)
print(li)
S_t, U_t = np.linalg.eigh(H)
print(S_t[0], S_t[-1])
