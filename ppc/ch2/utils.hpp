#include <bits/stdc++.h>

void fill_matrix(float* d, const int n) {
  for (int i = 0; i < n; ++i) {
    d[i * n + i] = 0.0;
    for (int j = i + 1; j < n; ++j) {
      float r = float(rand()) / RAND_MAX;
      d[i * n + j] = r;
      d[j * n + i] = r;
    }
  }
}