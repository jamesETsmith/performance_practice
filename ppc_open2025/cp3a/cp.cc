/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

int constexpr nb = 4; // size of block
using double4_t = double __attribute__((vector_size(nb * sizeof(double))));

constexpr double4_t d4zero = {0, 0, 0, 0};

static inline double reduce_sqrt(double4_t x) {
  double result = 0;
  for (int i = 0; i < 4; i++) {
    result += x[i];
  }
  return sqrt(result);
}

static inline double hdot4(double4_t a, double4_t b) {
  double result = 0;
  for (int i = 0; i < 4; i++) {
    result += a[i] * b[i];
  }
  return result;
}

void correlate(int ny, int nx, const float *data, float *result) {
  int na = (nx + nb - 1) / nb;
  std::vector<double4_t> X(ny * na, d4zero);

// standardize the rows (set the mean to 0 and std to 1)
#pragma omp parallel for
  for (int y = 0; y < ny; y++) {
    double mean = 0;
    for (int x = 0; x < nx; x++) {
      mean += data[x + y * nx];
    }
    mean /= nx;
    double std = 0;
    for (int x = 0; x < nx; x++) {
      std += (data[x + y * nx] - mean) * (data[x + y * nx] - mean);
    }
    // Use the population standard deviation because we are using the entire
    // dataset
    std = sqrt(std / nx);

    for (int x = 0; x < nx; x++) {
      X[x / nb + y * na][x % nb] = (data[x + y * nx] - mean) / std;
    }
  }

  // normalize the rows
#pragma omp parallel for
  for (int y = 0; y < ny; y++) {
    double4_t sum = d4zero;
    for (int xa = 0; xa < na; ++xa) {
      sum += X[xa + y * na] * X[xa + y * na];
    }
    double norm = reduce_sqrt(sum);
    for (int xa = 0; xa < na; ++xa) {
      for (int xb = 0; xb < nb; ++xb) {
        X[xa + y * na][xb] /= norm;
      }
    }
  }

  // compute XX^T
#pragma omp parallel for collapse(2)
  for (int j = 0; j < ny; ++j) {
    for (int i = j; i < ny; ++i) {

      // Break up the inner product into a sum of nb smaller inner products
      double res = 0;

      // Iterate over blocks of nb size
      for (int xa = 0; xa < na; ++xa) {
        // iterate over the block
        res += hdot4(X[xa + i * na], X[xa + j * na]);
      }

      result[i + j * ny] = res;
    }
  }
}
