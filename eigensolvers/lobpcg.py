import numpy as np
import scipy as sp

# Alg 1 in https://crd.lbl.gov/assets/Uploads/ieeetpds-mfdn-lobpcg-rev.pdf


def lobpcg_bad(
    H: np.ndarray,
    psi: np.ndarray,
    max_n_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    n_roots = psi.shape[1]

    # orthonormalize psi
    psi_orth = np.linalg.qr(psi)[0]
    print("psi_orth", psi_orth.shape)

    P = np.zeros_like(psi_orth)

    for i in range(max_n_iter):
        print(f"iter {i}")
        H_psi = np.einsum("ij,jk->ik", H, psi_orth)

        E_i = np.einsum("ij,ij->j", psi_orth.conj(), H_psi)
        R_i = H_psi - np.einsum("j,ij->ij", E_i, psi_orth)
        S_i = np.concatenate([psi_orth, R_i, P], axis=1)
        S_i_orth = np.linalg.qr(S_i)[0]

        # Rayleigh-Ritz
        shs = np.einsum("ij,ik,kl->jl", S_i_orth.conj(), H, S_i_orth)
        print(shs.shape)
        s, u = np.linalg.eigh(shs)
        print(s)
        psi_orth = S_i @ u[:, :n_roots]
        P = S_i @ u[:, n_roots:]
        # print(S_i.shape)


def lobpcg(A: np.ndarray, X: np.ndarray, max_iter: int = 25, tol: float = 1e-5):
    nx = X.shape[1]

    Th, C = np.linalg.eigh(np.einsum("ia,ij,jb->ab", X.conj(), A, X))
    Xi = np.einsum("ia,ab->ib", X, C)  # Rotate the basis of columns
    Ri = A @ Xi - np.einsum("ia,a->ia", Xi, Th)
    Pi = np.zeros_like(Xi)

    for i in range(max_iter):
        Wi = Ri  # this is where we'd apply the preconditioner
        Si = np.concatenate([Xi, Wi, Pi], axis=1)
        Si_orth = np.linalg.qr(Si)[0]
        shs = np.einsum("ia,ij,jb->ab", Si_orth.conj(), A, Si_orth)
        Thi, Ci = np.linalg.eigh(shs)
        print(f"iter {i} eigenvalues", Thi[:nx])

        Xi = np.einsum("ia,ab->ib", Si_orth, Ci[:, :nx])
        Ri = A @ Xi - np.einsum("ia,a->ia", Xi, Thi[:nx])
        Pi = np.einsum("ia,ab->ib", Si_orth[:, nx:], Ci[nx:, :nx])


np.random.seed(18)
N = 1000
nx = 10
H = np.random.rand(N, N)
H = H + H.T
psi = np.random.rand(N, nx)

np.set_printoptions(linewidth=160)

lobpcg(H, psi, max_iter=50)
S_t, U_t = np.linalg.eigh(H)
print(S_t[:nx])
