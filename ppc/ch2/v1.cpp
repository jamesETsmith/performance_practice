#include <iostream>
#include <limits>
#include <vector>

#include "utils.hpp"

void step(float *r, const float *d, int n) {
  std::vector<float> t(n * n);
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      t[n * j + i] = d[n * i + j];
    }
  }

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float v = std::numeric_limits<float>::infinity();

      asm("# loop over k start");
      for (int k = 0; k < n; ++k) {
        float x = d[n * i + k];
        float y = t[n * j + k];
        float z = x + y;
        v = std::min(v, z);
      }
      asm("# loop over k end");

      r[n * i + j] = v;
    }
  }
}

int main() {
  //   constexpr int n = 3;
  //   const float d[n * n] = {
  //       0, 8, 2, 1, 0, 9, 4, 5, 0,
  //   };

  const int n = 3000;
  std::vector<float> d(n * n);
  fill_matrix(d.data(), n);

  //   float r[n * n];
  std::vector<float> r(n * n);
  step(r.data(), d.data(), n);

  //   for (int i = 0; i < n; ++i) {
  //     for (int j = 0; j < n; ++j) {
  //       std::cout << r[i * n + j] << " ";
  //     }
  //     std::cout << "\n";
  //   }
}