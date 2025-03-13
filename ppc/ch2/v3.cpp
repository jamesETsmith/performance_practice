#include <iostream>
#include <limits>
#include <vector>

#include "utils.hpp"

typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));

constexpr float infty = std::numeric_limits<float>::infinity();

constexpr float8_t f8infty{infty, infty, infty, infty,
                           infty, infty, infty, infty};

static inline float hmin8(float8_t vv) {
  float v = infty;
  for (int i = 0; i < 8; ++i) {
    v = std::min(vv[i], v);
  }
  return v;
}

static inline float8_t min8(float8_t x, float8_t y) { return x < y ? x : y; }

void step(float *r, const float *d_, int n) {
  // elements per vector
  constexpr int nb = 8;
  // vectors per input row
  int na = (n + nb - 1) / nb;

  // input data, padded, converted to vectors
  std::vector<float8_t> vd(n * na);
  // input data, transposed, padded, converted to vectors
  std::vector<float8_t> vt(n * na);

#pragma omp parallel for
  for (int j = 0; j < n; ++j) {
    for (int ka = 0; ka < na; ++ka) {
      for (int kb = 0; kb < nb; ++kb) {
        int i = ka * nb + kb;
        vd[na * j + ka][kb] = i < n ? d_[n * j + i] : infty;
        vt[na * j + ka][kb] = i < n ? d_[n * i + j] : infty;
      }
    }
  }

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float8_t vv = f8infty;
      for (int ka = 0; ka < na; ++ka) {
        float8_t x = vd[na * i + ka];
        float8_t y = vt[na * j + ka];
        float8_t z = x + y;
        vv = min8(vv, z);
      }
      r[n * i + j] = hmin8(vv);
    }
  }
}

int main() {

  const int n = 3000;
  std::vector<float> d(n * n);
  fill_matrix(d.data(), n);

  //   float r[n * n];
  std::vector<float> r(n * n);
  step(r.data(), d.data(), n);
}