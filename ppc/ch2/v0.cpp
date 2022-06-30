#include <iostream>
#include <limits>
#include <vector>

#include "utils.hpp"

asm("# foo");
void step(float* r, const float* d, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float v = std::numeric_limits<float>::infinity();
      for (int k = 0; k < n; ++k) {
        float x = d[n * i + k];
        float y = d[n * k + j];
        float z = x + y;
        v = std::min(v, z);
      }
      r[n * i + j] = v;
    }
  }
}
asm("# foo");

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