#include <bits/stdc++.h>

#include <iostream>

#include "ubench.h"

//
// User settings
//

const int N = 1920;

//
// Machine Specific Constants
//

// User specified
// Use lscpu
const int N_PROCS = 6;
const int L1 = 192000;
const int L2 = 1572864;
const int L3 = 9437184;

// Automatically calculated
const int B = 8;  // number of elements in a vector
typedef float vec __attribute__((vector_size(4 * B)));

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

  // Dynamically calculated s values
  // how many columns of b fit in L3
  // const int s3 = std::min(L3 / nx / 16 * 16, ny);
  // // how many rows of a fit in L2
  // const int s2 = std::min(L2 / ny / 6 * 6, nx);
  // // how tall a (k x s3) block in b can be to fit in L1
  // const int s1 = std::min(L1 / s3, nx);
  // std::cout << "S3 " << s3 << "\nS2 " << s2 << "\nS1 " << s1 << std::endl;

  // Original code below:
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
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v4(a, b, c, N); }
}

//
// V6 Unrolled
//
// c: 6 x 16
// a: 6 x k
// b: k x 16
// c[x:x+6][y:y+16] += a[x:x+6][l:r] * b[l:r][y:y+16]

void kernel_v5(float *a, vec *b, vec *c, int x, int y, int l, int r, int n) {
  vec t00, t01, t10, t11, t20, t21, t30, t31, t40, t41, t50, t51;

  t00 = c[((x + 0) * n + y) / 8 + 0];
  t01 = c[((x + 0) * n + y) / 8 + 1];

  t10 = c[((x + 1) * n + y) / 8 + 0];
  t11 = c[((x + 1) * n + y) / 8 + 1];

  t20 = c[((x + 2) * n + y) / 8 + 0];
  t21 = c[((x + 2) * n + y) / 8 + 1];

  t30 = c[((x + 3) * n + y) / 8 + 0];
  t31 = c[((x + 3) * n + y) / 8 + 1];

  t40 = c[((x + 4) * n + y) / 8 + 0];
  t41 = c[((x + 4) * n + y) / 8 + 1];

  t50 = c[((x + 5) * n + y) / 8 + 0];
  t51 = c[((x + 5) * n + y) / 8 + 1];

  for (int k = l; k < r; k++) {
    vec a0 = vec{} + a[(x + 0) * n + k];
    t00 += a0 * b[(k * n + y) / 8];
    t01 += a0 * b[(k * n + y) / 8 + 1];

    vec a1 = vec{} + a[(x + 1) * n + k];
    t10 += a1 * b[(k * n + y) / 8];
    t11 += a1 * b[(k * n + y) / 8 + 1];

    vec a2 = vec{} + a[(x + 2) * n + k];
    t20 += a2 * b[(k * n + y) / 8];
    t21 += a2 * b[(k * n + y) / 8 + 1];

    vec a3 = vec{} + a[(x + 3) * n + k];
    t30 += a3 * b[(k * n + y) / 8];
    t31 += a3 * b[(k * n + y) / 8 + 1];

    vec a4 = vec{} + a[(x + 4) * n + k];
    t40 += a4 * b[(k * n + y) / 8];
    t41 += a4 * b[(k * n + y) / 8 + 1];

    vec a5 = vec{} + a[(x + 5) * n + k];
    t50 += a5 * b[(k * n + y) / 8];
    t51 += a5 * b[(k * n + y) / 8 + 1];
  }

  c[((x + 0) * n + y) / 8 + 0] = t00;
  c[((x + 0) * n + y) / 8 + 1] = t01;

  c[((x + 1) * n + y) / 8 + 0] = t10;
  c[((x + 1) * n + y) / 8 + 1] = t11;

  c[((x + 2) * n + y) / 8 + 0] = t20;
  c[((x + 2) * n + y) / 8 + 1] = t21;

  c[((x + 3) * n + y) / 8 + 0] = t30;
  c[((x + 3) * n + y) / 8 + 1] = t31;

  c[((x + 4) * n + y) / 8 + 0] = t40;
  c[((x + 4) * n + y) / 8 + 1] = t41;

  c[((x + 5) * n + y) / 8 + 0] = t50;
  c[((x + 5) * n + y) / 8 + 1] = t51;
}

/*
const int L1 = (1<<15) / 4; // L1 cache is 32K
const int L2 = (1<<19) / 4; // L2 cache is 512K
const int L3 = (1<<23) / 4; // L3 cache is 8M
*/

void matmul_v5(const float *_a, const float *_b, float *_c, int n) {
  int nx = (n + 5) / 6 * 6;
  int ny = (n + 15) / 16 * 16;

  const int MAXN = N * N;  // ~15MB each
  alignas(64) static float a[MAXN], b[MAXN], c[MAXN];

  /*for (int i = 0; i < n; i++) {
      memcpy(&a[i * ny], &_a[i * n], 4 * n);
      memcpy(&b[i * ny], &_b[i * n], 4 * n);
  }*/

  // c[x:x+6][y:y+16] += a[x:x+6][l:r] * b[l:r][y:y+16]

  // load b[i*L1 : (i+1)*L1][y:y+16] into L1 cache and iterate over a
  // when out of L2 cache to hold a, load new strip of b and continue
  // when out of L3 cache to hold b, switch to new segment of a

  // divide b into segments that fit L3, fix a segment
  // divide a into segments that fit L2, fix a segment
  // divide b into segments that fit L1, fix a segment
  // iterate over a

  /*
  // how many columns of b fit in L3
  const int s3 = std::min(L3 / nx / 16 * 16, ny);
  // how many rows of a fit in L2
  const int s2 = std::min(L2 / ny / 6 * 6, nx);
  // how tall a (k x s3) block in b can be to fit in L1
  const int s1 = std::min(L1 / s3, nx);
  */

  // s3 * nx < L3 (doesn't really matter)
  // s2 * ny < L2
  // s1 * s3 < L1
  // s1 -> max

  // const int s1 = std::min(L1 / 16, nx);
  // const int s2 = L2 / ny / 6 * 6;
  // const int s3 = 16;

  const int s3 = 64;
  const int s2 = 120;
  const int s1 = 240;

  /*
  const int u = 96;
  const int s3 = u;
  const int s2 = 2 * u;
  const int s1 = 4 * u;
  */

  // const int t = L1/s3;

  // 1 252 4032
  // std::cerr << s1 << " " << s2 << " " << s3 << std::endl;

  for (int i3 = 0; i3 < ny; i3 += s3)
    // now we are working with b[:][i3:i3+s3]
    for (int i2 = 0; i2 < nx; i2 += s2)
      // now we are working with a[i2:i2+s2][:]
      for (int i1 = 0; i1 < ny; i1 += s1)
        // now we are working with b[i1:i1+s1][i3:i3+s3]
        // this equates to updating c[i2:i2+s2][i3:i3+s3]
        // with [l:r] = [i1:i1+s1]
        for (int x = i2; x < i2 + s2; x += 6)
          for (int y = i3; y < i3 + s3; y += 16)
            kernel_v5(a, (vec *)b, (vec *)c, x, y, i1, i1 + s1, ny);

  // for (int i = 0; i < n; i++)
  //     memcpy(&_c[i * n], &c[i * ny], 4 * n);
}

UBENCH_EX(matmul, v5) {
  float *a = new float[N * N];
  float *b = new float[N * N];
  float *c = new float[N * N];

  _fill(a, b, c, N * N);

  UBENCH_DO_BENCHMARK() { matmul_v5(a, b, c, N); }
}

//
//
//

UBENCH_MAIN();