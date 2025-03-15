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

int constexpr nb = 4; // size of vector
using double4_t = double __attribute__((vector_size(nb * sizeof(double))));

constexpr double4_t d4zero = {0, 0, 0, 0};

static inline double reduce_sqrt(double4_t x) {
  double result = 0;
  for (int i = 0; i < 4; i++) {
    result += x[i];
  }
  return sqrt(result);
}

static inline double reduce_d4t(double4_t x) {
  double result = 0;
  for (int i = 0; i < 4; i++) {
    result += x[i];
  }
  return result;
}

void correlate(int ny, int nx, const float *data, float *result) {
  // Calculate the number of vectors we'll need for each row
  int na = (nx + nb - 1) / nb;

  // size of blocks that we'll update in output
  int constexpr nd = 3;
  // number of blocks we'll need for each column
  int nc = (ny + nd - 1) / nd;
  // Number of rows after padding
  int ncd = nc * nd;

  std::vector<double4_t> X(ncd * na, d4zero);

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
#pragma omp parallel for schedule(dynamic)
  for (int jc = 0; jc < nc; ++jc) {
    for (int ic = jc; ic < nc; ++ic) {

      std::array<std::array<double4_t, nd>, nd> vv = {d4zero};

      for (int ka = 0; ka < na; ++ka) {
        // indices are a mess here
        // (unpacked row idx) * na + vector idx
        double4_t x0 = X[(jc * nd + 0) * na + ka];
        double4_t x1 = X[(jc * nd + 1) * na + ka];
        double4_t x2 = X[(jc * nd + 2) * na + ka];
        double4_t y0 = X[(ic * nd + 0) * na + ka];
        double4_t y1 = X[(ic * nd + 1) * na + ka];
        double4_t y2 = X[(ic * nd + 2) * na + ka];

        vv[0][0] += x0 * y0;
        vv[0][1] += x0 * y1;
        vv[0][2] += x0 * y2;
        vv[1][0] += x1 * y0;
        vv[1][1] += x1 * y1;
        vv[1][2] += x1 * y2;
        vv[2][0] += x2 * y0;
        vv[2][1] += x2 * y1;
        vv[2][2] += x2 * y2;
      }

      // reduce vv and store in result
      for (int id = 0; id < nd; ++id) {
        for (int jd = 0; jd < nd; ++jd) {
          int i = ic * nd + id;
          int j = jc * nd + jd;

          if (i < ny && j < ny) {
            result[i + j * ny] = reduce_d4t(vv[jd][id]);
          }
        }
      }
    }
  }
}
