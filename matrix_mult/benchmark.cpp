#include <bits/stdc++.h>

#include "ubench.h"

//
// Utils
//
void _fill(float *a, float *b, float *c, const int NN) {
  for (int i = 0; i < NN; i++) {
    a[i] = float(rand()) / RAND_MAX;
    b[i] = float(rand()) / RAND_MAX;
    c[i] = 0;
  }
}

typedef float vec __attribute__((vector_size(32)));

// a helper function that allocates n vectors and initializes them with zeros
vec *alloc(int n) {
  vec *ptr = (vec *)std::aligned_alloc(32, 32 * n);
  memset(ptr, 0, 32 * n);
  return ptr;
}

float *alloc_float(int n) {
  float *ptr = (float *)std::aligned_alloc(32, 32 * n);
  memset(ptr, 0, 32 * n);
  return ptr;
}

//
// V0: Naive Implementation
//

void matmul_v0(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++) c[i * n + j] += a[i * n + k] * b[k * n + j];
}

UBENCH_EX(matmul, v0) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v0(a, b, c, N); }
}

//
// V1: Transpose b
//
void matmul_v1(const float *a, const float *_b, float *c, int n) {
  float *b = new float[n * n];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) b[i * n + j] = _b[j * n + i];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        c[i * n + j] += a[i * n + k] * b[j * n + k];  // <- note the indices
}

UBENCH_EX(matmul, v1) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v1(a, b, c, N); }
}

//
// V2 vectorization
//

// Do the vectorization manually
void matmul_v2_manual(const float *_a, const float *_b, float *c, int n) {
  int nB = (n + 7) / 8;  // number of 8-element vectors in a row (rounded up)

  vec *a = alloc(n * nB);
  vec *b = alloc(n * nB);

  // move both matrices to the aligned region
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i * nB + j / 8][j % 8] = _a[i * n + j];
      b[i * nB + j / 8][j % 8] = _b[j * n + i];  // <- b is still transposed
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      vec s{};  // initialize the accumulator with zeros

      // vertical summation
      for (int k = 0; k < nB; k++) s += a[i * nB + k] * b[j * nB + k];

      // horizontal summation
      for (int k = 0; k < 8; k++) c[i * n + j] += s[k];
    }
  }

  std::free(a);
  std::free(b);
}

UBENCH_EX(matmul, v2_manual) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v2_manual(a, b, c, N); }
}

// Let the compiler do the vectoriation
void matmul_v2(const float *a, const float *_b, float *__restrict__ c, int n) {
  float *b = new float[n * n];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) b[i * n + j] = _b[j * n + i];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float s = 0;
      for (int k = 0; k < n; k++) s += a[i * n + k] * b[j * n + k];
      c[i * n + j] = s;
    }
  }
}

UBENCH_EX(matmul, v2) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v2(a, b, c, N); }
}

//
// V3 Kernel: Update a submatrix to reduce memory IO (key concept:
// https://en.algorithmica.org/hpc/algorithms/matmul/#register-reuse)
//

// update 6x16 submatrix C[x:x+6][y:y+16]
// using A[x:x+6][l:r] and B[l:r][y:y+16]
void kernel_v3(float *a, vec *b, vec *c, int x, int y, int l, int r, int n) {
  vec t[6][2]{};  // will be zero-filled and stored in ymm registers

  for (int k = l; k < r; k++) {
    for (int i = 0; i < 6; i++) {
      // broadcast a[x + i][k] into a register
      vec alpha = vec{} + a[(x + i) * n + k];  // converts to a broadcast
      // multiply b[k][y:y+16] by it and update t[i][0] and t[i][1]
      for (int j = 0; j < 2; j++)
        t[i][j] += alpha * b[(k * n + y) / 8 + j];  // converts to an fma
    }
  }

  // write the results back to C
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 2; j++) c[((x + i) * n + y) / 8 + j] += t[i][j];
}

void matmul_v3(const float *_a, const float *_b, float *_c, int n) {
  // to simplify the implementation, we pad the height and width
  // so that they are divisible by 6 and 16 respectively
  int nx = (n + 5) / 6 * 6;
  int ny = (n + 15) / 16 * 16;

  float *a = alloc_float(nx * ny);
  float *b = alloc_float(nx * ny);
  float *c = alloc_float(nx * ny);

  for (int i = 0; i < n; i++) {
    memcpy(&a[i * ny], &_a[i * n], 4 * n);
    memcpy(&b[i * ny], &_b[i * n],
           4 * n);  // we don't need to transpose b this time
  }

  for (int x = 0; x < nx; x += 6)
    for (int y = 0; y < ny; y += 16)
      kernel_v3(a, (vec *)b, (vec *)c, x, y, 0, n, ny);

  for (int i = 0; i < n; i++) memcpy(&_c[i * n], &c[i * ny], 4 * n);

  std::free(a);
  std::free(b);
  std::free(c);
}

UBENCH_EX(matmul, v3) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v3(a, b, c, N); }
}

//
// V4: Using blocking
//

// From: https://en.algorithmica.org/hpc/algorithms/matmul/#blocking
// Cache blocking is less trivial to do with matrices than with arrays, but the
// general idea is this:

// 1) Select a submatrix of B that fits into the L3 cache (say, a subset of its
// columns).
//
// 2) Select a submatrix of A that fits into the L2 cache (say, a subset
// of its rows).
//
// 3) Select a submatrix of the previously selected submatrix of B (a subset of
// its rows) that fits into the L1 cache.
//
// 4) Update the relevant submatrix of C using the kernel.

void matmul_v4(const float *_a, const float *_b, float *_c, int n) {
  // to simplify the implementation, we pad the height and width
  // so that they are divisible by 6 and 16 respectively
  int nx = (n + 5) / 6 * 6;
  int ny = (n + 15) / 16 * 16;

  float *a = alloc_float(nx * ny);
  float *b = alloc_float(nx * ny);
  float *c = alloc_float(nx * ny);

  for (int i = 0; i < n; i++) {
    memcpy(&a[i * ny], &_a[i * n], 4 * n);
    memcpy(&b[i * ny], &_b[i * n],
           4 * n);  // we don't need to transpose b this time
  }

  const int s3 = 64;   // how many columns of B to select
  const int s2 = 120;  // how many rows of A to select
  const int s1 = 240;  // how many rows of B to select

  for (int i3 = 0; i3 < ny; i3 += s3)
    // now we are working with b[:][i3:i3+s3]
    for (int i2 = 0; i2 < nx; i2 += s2)
      // now we are working with a[i2:i2+s2][:]
      for (int i1 = 0; i1 < ny; i1 += s1)
        // now we are working with b[i1:i1+s1][i3:i3+s3]
        // and we need to update c[i2:i2+s2][i3:i3+s3] with [l:r] = [i1:i1+s1]
        for (int x = i2; x < std::min(i2 + s2, nx); x += 6)
          for (int y = i3; y < std::min(i3 + s3, ny); y += 16)
            kernel_v3(a, (vec *)b, (vec *)c, x, y, i1, std::min(i1 + s1, n),
                      ny);

  std::free(a);
  std::free(b);
  std::free(c);
}

UBENCH_EX(matmul, v4) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v4(a, b, c, N); }
}

//
//
//

UBENCH_MAIN();