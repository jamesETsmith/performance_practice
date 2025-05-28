#include <iostream>
#include <limits>
#include <vector>

#include "utils.hpp"

typedef float float16_t __attribute__((vector_size(16 * sizeof(float))));

constexpr float infty = std::numeric_limits<float>::infinity();

constexpr float16_t f16infty{infty, infty, infty, infty, infty, infty,
                             infty, infty, infty, infty, infty, infty,
                             infty, infty, infty, infty};

static inline float hmin16(float16_t vv) {
  float v = infty;
  for (int i = 0; i < 16; ++i) {
    v = std::min(vv[i], v);
  }
  return v;
}

static inline float16_t min16(float16_t x, float16_t y) {
  return x < y ? x : y;
}

void step(float *r, const float *d_, int n) {
  // elements per vector
  constexpr int nb = 16;
  // vectors per input row
  int na = (n + nb - 1) / nb;

  // input data, padded, converted to vectors
  std::vector<float16_t> vd(n * na);
  // input data, transposed, padded, converted to vectors
  std::vector<float16_t> vt(n * na);

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
      float16_t vv = f16infty;
      asm("# loop over k start");
      for (int ka = 0; ka < na; ++ka) {
        float16_t x = vd[na * i + ka];
        float16_t y = vt[na * j + ka];
        float16_t z = x + y;
        vv = min16(vv, z);
      }
      asm("# loop over k end");
      r[n * i + j] = hmin16(vv);
    }
  }
}

int main() {

  const int n = 3000;
  std::vector<float> d(n * n);
  fill_matrix(d.data(), n);

  std::vector<float> r(n * n);
  step(r.data(), d.data(), n);
}