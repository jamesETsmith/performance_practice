import time
import numpy as np
import scipy as sp


def lanczos_iter(
    A: np.ndarray, b: np.ndarray, max_iter: int = 10, tol: float = 1e-5
) -> np.ndarray:
    T_n = np.zeros((max_iter, max_iter))
    Q_n = np.zeros((A.shape[0], max_iter), order="F")

    Q_n[:, 0] = b / np.linalg.norm(b)
    beta_n1 = 0

    for n in range(max_iter - 1):
        v = np.einsum("ij,j", A, Q_n[:, n])
        alpha = np.dot(Q_n[:, n].conj(), v)
        v = v - beta_n1 * Q_n[:, n - 1] - alpha * Q_n[:, n]
        beta = np.linalg.norm(v)
        Q_n[:, n + 1] = v / beta

        T_n[n, n] = alpha
        T_n[n, n + 1] = beta
        T_n[n + 1, n] = beta

        beta_n1 = beta

    # Final iteration
    v = np.einsum("ij,j", A, Q_n[:, max_iter - 1])
    alpha = np.dot(Q_n[:, max_iter - 1].conj(), v)
    T_n[max_iter - 1, max_iter - 1] = alpha

    S, U = np.linalg.eigh(T_n)

    Q_n = Q_n @ U
    eig_val = S[0]
    eig_vec = Q_n[:, 0]

    return eig_val, eig_vec


def lanczos_iter_batch(
    A: np.ndarray, b: np.ndarray, max_iter: int = 10, tol: float = 1e-5
) -> np.ndarray:
    batch_size = A.shape[0]
    dim = A.shape[1]

    T_n = np.zeros((batch_size, max_iter, max_iter))  # (b, n, n)
    Q_n = np.zeros((batch_size, max_iter, dim))  # (b, n, i)
    Q_n[:, 0, :] = b / np.linalg.norm(b, axis=1)[:, None]
    beta_n1 = np.zeros(batch_size)

    for n in range(max_iter - 1):
        v = np.einsum("bij,bj->bi", A, Q_n[:, n, :])
        alpha = np.einsum("bi,bi->b", Q_n[:, n, :], v)
        v = v - beta_n1[:, None] * Q_n[:, n - 1, :] - alpha[:, None] * Q_n[:, n, :]
        beta = np.linalg.norm(v, axis=1)
        Q_n[:, n + 1, :] = v / beta[:, None]

        T_n[:, n, n] = alpha
        T_n[:, n, n + 1] = beta
        T_n[:, n + 1, n] = beta

        beta_n1 = beta

    # Final iteration
    v = np.einsum("bij,bj->bi", A, Q_n[:, max_iter - 1, :])
    alpha = np.einsum("bi,bi->b", Q_n[:, max_iter - 1, :], v)
    T_n[:, max_iter - 1, max_iter - 1] = alpha

    S, U = np.linalg.eigh(T_n)

    Q_n = np.einsum("bni,bnm->bmi", Q_n, U)
    eig_val = S[:, 0]
    eig_vec = Q_n[:, 0, :]
    print(eig_val.shape, eig_vec.shape)

    return eig_val, eig_vec


def lanczos_iter_batch_tridiag(
    A: np.ndarray, b: np.ndarray, max_iter: int = 10, tol: float = 1e-5
) -> np.ndarray:
    batch_size = A.shape[0]
    dim = A.shape[1]

    # T_n = np.zeros((batch_size, max_iter, max_iter))  # (b, n, n)
    diag = np.zeros((batch_size, max_iter))  # (b, n)
    sub_diag = np.zeros((batch_size, max_iter - 1))  # (b, n-1)
    Q_n = np.zeros((batch_size, max_iter, dim))  # (b, n, i)
    Q_n[:, 0, :] = b / np.linalg.norm(b, axis=1)[:, None]
    beta_n1 = np.zeros(batch_size)

    for n in range(max_iter - 1):
        v = np.einsum("bij,bj->bi", A, Q_n[:, n, :])
        alpha = np.einsum("bi,bi->b", Q_n[:, n, :], v)
        v = v - beta_n1[:, None] * Q_n[:, n - 1, :] - alpha[:, None] * Q_n[:, n, :]
        beta = np.linalg.norm(v, axis=1)
        Q_n[:, n + 1, :] = v / beta[:, None]

        diag[:, n] = alpha
        sub_diag[:, n] = beta

        beta_n1 = beta

    # Final iteration
    v = np.einsum("bij,bj->bi", A, Q_n[:, max_iter - 1, :])
    alpha = np.einsum("bi,bi->b", Q_n[:, max_iter - 1, :], v)
    diag[:, max_iter - 1] = alpha

    S = np.zeros((batch_size, max_iter))
    U = np.zeros((batch_size, max_iter, max_iter))
    for i in range(batch_size):
        S[i, :], U[i, :, :] = sp.linalg.eigh_tridiagonal(diag[i, :], sub_diag[i, :])

    Q_n = np.einsum("bni,bnm->bmi", Q_n, U)
    eig_val = S[:, 0]
    eig_vec = Q_n[:, 0, :]
    print(eig_val.shape, eig_vec.shape)

    return eig_val, eig_vec


def test_lanczos_single():
    #
    np.random.seed(18)

    N = 100
    A = np.random.rand(N, N)
    A = A + A.T
    b = np.random.rand(N)

    t0 = time.perf_counter()
    S0, U0 = lanczos_iter(A, b, max_iter=50)
    print(f"Time Lanczos: {time.perf_counter() - t0:.2e}")
    print(S0)
    # print(U0)

    t0 = time.perf_counter()
    S, U = np.linalg.eigh(A)
    print(f"Time eigh: {time.perf_counter() - t0:.2e}")
    print(S[0])
    # print(U[:, 0])

    ovlp = np.abs(np.einsum("i,i", U0, U[:, 0]))
    np.testing.assert_allclose(S0, S[0], rtol=1e-6)
    np.testing.assert_allclose(ovlp, 1, rtol=1e-5)


def test_lanczos_batch():
    #
    np.random.seed(18)

    N = 128
    batch_size = 1000
    A = np.random.rand(batch_size, N, N)
    A = A + A.transpose(0, 2, 1)
    A[:, np.arange(N), np.arange(N)] = np.random.rand(batch_size, N)
    b = np.random.rand(batch_size, N)

    t0 = time.perf_counter()
    S0, U0 = lanczos_iter_batch(A, b, max_iter=max(N // 10, 60))
    print(f"Time batch Lanczos: {time.perf_counter() - t0:.2e}")

    t0 = time.perf_counter()
    S, U = np.linalg.eigh(A)
    print(f"Time batch eigh: {time.perf_counter() - t0:.2e}")

    ovlp = np.abs(np.einsum("bi,bi->b", U0, U[:, :, 0]))

    np.testing.assert_allclose(S0, S[:, 0], rtol=1e-6)
    np.testing.assert_allclose(ovlp, np.ones(batch_size), rtol=1e-5)


def test_lanczos_batch_tridiag():
    #
    np.random.seed(18)

    N = 128
    batch_size = 1000
    A = np.random.rand(batch_size, N, N)
    A = A + A.transpose(0, 2, 1)
    A[:, np.arange(N), np.arange(N)] = np.random.rand(batch_size, N)
    b = np.random.rand(batch_size, N)

    t0 = time.perf_counter()
    S0, U0 = lanczos_iter_batch_tridiag(A, b, max_iter=max(N // 10, 60))
    print(f"Time batch tridiagonal Lanczos: {time.perf_counter() - t0:.2e}")

    t0 = time.perf_counter()
    S, U = np.linalg.eigh(A)
    print(f"Time batch eigh:                {time.perf_counter() - t0:.2e}")

    ovlp = np.abs(np.einsum("bi,bi->b", U0, U[:, :, 0]))

    np.testing.assert_allclose(S0, S[:, 0], rtol=1e-6)
    np.testing.assert_allclose(ovlp, np.ones(batch_size), rtol=1e-5)


def test_bench_lanczos_batch(benchmark):
    N = 1 << 12
    batch_size = 10
    A = np.random.rand(batch_size, N, N)
    A = A + A.transpose(0, 2, 1)
    A[:, np.arange(N), np.arange(N)] = np.random.rand(batch_size, N)
    b = np.random.rand(batch_size, N)

    benchmark(lanczos_iter_batch, A, b, max_iter=max(N // 10, 60))


def test_bench_lanczos_batch_tridiag(benchmark):
    N = 1 << 12
    batch_size = 10
    A = np.random.rand(batch_size, N, N)
    A = A + A.transpose(0, 2, 1)
    A[:, np.arange(N), np.arange(N)] = np.random.rand(batch_size, N)
    b = np.random.rand(batch_size, N)

    benchmark(lanczos_iter_batch_tridiag, A, b, max_iter=max(N // 10, 60))
