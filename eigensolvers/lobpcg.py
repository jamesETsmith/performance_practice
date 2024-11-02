import numpy as np
import scipy as sp
import pytest
import torch
from typing import Callable


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


def batch_lobpcg(
    A: torch.Tensor, X: torch.Tensor, max_iter: int = 50, tol: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    n_batches = X.shape[0]
    dim = X.shape[1]
    nx = X.shape[2]

    if nx > dim:
        raise ValueError(
            "Number of eigenvectors requested is greater than the dimension of the space"
        )

    # Solve initial rayleigh-ritz
    Th, C = torch.linalg.eigh(torch.einsum("tia,tij,tjb->tab", X.conj(), A, X))

    # Rotate the appro eigenvectors
    Xi = torch.einsum("tia,tab->tib", X, C)

    # Residuals
    Ri = torch.einsum("tij,tja->tia", A, Xi) - torch.einsum("tia,ta->tia", Xi, Th)

    # Previous approximate eigenvectors
    Pi = torch.zeros_like(Xi)

    for i in range(max_iter):
        Wi = Ri
        Si = torch.cat([Xi, Wi, Pi], dim=2)
        Si_orth = torch.linalg.qr(Si)[0]  # (n_batches, , 3*nx)

        shs = torch.einsum("tia,tij,tjb->tab", Si_orth.conj(), A, Si_orth)
        Thi, Ci = torch.linalg.eigh(shs)

        Xi = torch.einsum("tia,tab->tib", Si_orth, Ci[:, :, :nx])
        Ri = torch.einsum("tij,tja->tia", A, Xi) - torch.einsum(
            "tia,ta->tia", Xi, Thi[:, :nx]
        )

        if (torch.linalg.norm(Ri, axis=(1, 2)) < tol).all():
            print(f"Converged in {i} iterations")
            break

        Pi = torch.einsum("tia,tab->tib", Si_orth[:, :, nx:], Ci[:, nx:, :nx])

    return Thi[:, :nx], Xi


class BatchLinearOperator:
    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Callable[[torch.Tensor], torch.Tensor] = None,
        matmat: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        dtype: torch.dtype = None,
    ):
        if matvec is None and matmat is None:
            raise ValueError("Either matvec or matmat must be provided")

        self.shape = shape
        self.matvec = matvec
        self.matmat = matmat
        self.dtype = dtype


@pytest.mark.parametrize("n_batches", [1, 2, 10], ids=lambda x: f"nb={x}")
@pytest.mark.parametrize("nx", [1, 2, 16], ids=lambda x: f"nx={x}")
@pytest.mark.parametrize("N", [16, 32, 64, 128, 256], ids=lambda x: f"N={x}")
def test_batch_lobpcg(n_batches: int, nx: int, N: int) -> None:
    torch.manual_seed(18)
    torch.cuda.manual_seed(18)
    torch.set_default_dtype(torch.float64)
    # torch.set_default_device(torch.device("cuda"))

    H = torch.rand(n_batches, N, N)
    H = H + H.transpose(2, 1)
    psi = torch.rand(n_batches, N, nx)

    Thi, Xi = batch_lobpcg(H, psi, max_iter=2 * N, tol=1e-5)
    S_t, U_t = torch.linalg.eigh(H)

    ev_error = torch.abs(Thi - S_t[:, :nx])
    print(f"Eigenvalue error {ev_error}")
    torch.testing.assert_close(Thi, S_t[:, :nx], atol=1e-5, rtol=1e-5)

    fidelity = torch.abs(torch.einsum("tia,tia->ta", Xi.conj(), U_t[:, :, :nx]))
    print(f"Fidelity of eigenvectors {fidelity}")
    torch.testing.assert_close(
        fidelity, torch.ones_like(fidelity), atol=1e-5, rtol=1e-5
    )


if __name__ == "__main__":
    np.set_printoptions(linewidth=160)

    # np.random.seed(18)
    # N = 128
    # nx = 10
    # H = np.random.rand(N, N)
    # H = H + H.T
    # psi = np.random.rand(N, nx)

    # lobpcg(H, psi, max_iter=50)
    # S_t, U_t = np.linalg.eigh(H)
    # print(S_t[:nx])

    torch.manual_seed(18)
    N = 128
    nx = 1
    n_batches = 10

    H = torch.rand(n_batches, N, N)
    H = H + H.transpose(2, 1)
    psi = torch.rand(n_batches, N, nx)

    Thi, Xi = batch_lobpcg(H, psi, max_iter=50)
    S_t, U_t = torch.linalg.eigh(H)

    print(f"LOBPCG eigenvalues {Thi}")
    print(f"Eigenvalues {S_t[:,:nx]}")

    fidelity = torch.abs(torch.einsum("tia,tia->t", Xi, U_t[:, :, :nx]))
    print(f"Fidelity of eigenvectors {fidelity}")
