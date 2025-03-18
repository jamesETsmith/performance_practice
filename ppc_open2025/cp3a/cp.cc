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
#include <vector>

//
// User defined constants
//
int constexpr nb = 4; // size of vector
int constexpr nd = 8; // size of blocks that we'll update in output

//
// User defined types
//
using double4_t = double __attribute__((vector_size(nb * sizeof(double))));

constexpr double4_t d4zero = {0};

//
// Helper functions
//
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

  // number of blocks we'll need for each column
  int nc = (ny + nd - 1) / nd;
  // Number of rows after padding
  int ncd = nc * nd;

  // copy the data to padded vector array
  std::vector<double4_t> X(ncd * na, d4zero);

#pragma omp parallel
  {

#pragma omp for collapse(2)
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        X[x / nb + y * na][x % nb] = data[x + y * nx];
      }
    }

    // standardize the rows (set the mean to 0 and std to 1)

#pragma omp for
    for (int y = 0; y < ny; y++) {
      // mean
      double4_t mean_vec = d4zero;
      for (int xa = 0; xa < na; xa++) {
        mean_vec += X[xa + y * na];
      }
      // reduce and broadcast results
      double mean = reduce_d4t(mean_vec) / nx;
      for (int xb = 0; xb < nb; xb++) {
        mean_vec[xb] = mean;
      }

      // std
      double4_t std_vec = d4zero;
      for (int xa = 0; xa < na; xa++) {
        double4_t tmp = X[xa + y * na] - mean_vec;
        std_vec += tmp * tmp;
      }
      // reduce and broadcast results
      double std = sqrt(reduce_d4t(std_vec) / nx);
      for (int xb = 0; xb < nb; xb++) {
        std_vec[xb] = std;
      }

      // scale and calculate norm
      double4_t norm_vec = d4zero;
      for (int xa = 0; xa < ((nx % nb == 0) ? na : na - 1); xa++) {
        int idx = xa + y * na;
        X[idx] = (X[idx] - mean_vec) / std_vec;
        norm_vec += X[idx] * X[idx];
      }

      // for the tail we only populate the non-padded elements
      for (int xb = 0; xb < nx % nb; xb++) {
        int idx = na - 1 + y * na;
        X[idx][xb] = (X[idx][xb] - mean_vec[xb]) / std_vec[xb];
        norm_vec[xb] += X[idx][xb] * X[idx][xb];
      }

      double norm = reduce_sqrt(norm_vec);
      for (int xb = 0; xb < nb; xb++) {
        norm_vec[xb] = norm;
      }

      for (int xa = 0; xa < na; ++xa) {
        X[xa + y * na] /= norm_vec;
      }
    }

    // compute XX^T
#pragma omp for schedule(dynamic)
    for (int jc = 0; jc < nc; ++jc) {
      for (int ic = jc; ic < nc; ++ic) {

        std::array<std::array<double4_t, nd>, nd> vv = {d4zero};
        std::array<double4_t, nd> xi;
        std::array<double4_t, nd> yi;

        for (int ka = 0; ka < na; ++ka) {
          // indices are a mess here
          // (unpacked row idx) * na + vector idx

          for (int i = 0; i < nd; ++i) {
            xi[i] = X[(jc * nd + i) * na + ka];
            yi[i] = X[(ic * nd + i) * na + ka];
          }

          for (int i = 0; i < nd; ++i) {
            for (int j = 0; j < nd; ++j) {
              vv[i][j] += xi[i] * yi[j];
            }
          }
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
}
