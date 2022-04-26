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

//
//
//

void matmul_v0(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++) c[i * n + j] += a[i * n + k] * b[k * n + j];
}

void matmul_v1(const float *a, const float *_b, float *c, int n) {
  float *b = new float[n * n];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) b[i * n + j] = _b[j * n + i];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        c[i * n + j] += a[i * n + k] * b[j * n + k];  // <- note the indices
}

void matmul_v2(const float *_a, const float *_b, float *c, int n) {
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

void matmul_v3(const float *_a, const float *_b, float *__restrict__ c, int n) {
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

UBENCH_EX(matmul, v0) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v0(a, b, c, N); }
}

UBENCH_EX(matmul, v1) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v1(a, b, c, N); }
}

UBENCH_EX(matmul, v2) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v2(a, b, c, N); }
}

UBENCH_EX(matmul, v3) {
  const int N = 1920;
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v3(a, b, c, N); }
}

UBENCH_MAIN();