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
#include <cstring>
#include <immintrin.h>
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

/*
// update 6x16 submatrix C[x:x+6][y:y+16]
// using A[x:x+6][l:r] and B[l:r][y:y+16]
*/
void kernel(float *result, double const *X, int _x, int _y, int const nx,
            int const ny) {
  // double4_t C[6][4]{}; // zero filled and stored in ymm registers
  double C[6][16]{};

  for (int x = _x; x < std::min(ny, _x + 6); ++x) {
    for (int y = _y; y < std::min(ny, _y + 16); ++y) {

      // Always doing the whole row
      for (int k = 0; k < nx; ++k) {
        C[x - _x][y - _y] += X[k + x * nx] * X[k + y * nx];
      }
    }
  }

  // copy C to the results
  for (int x = _x; x < std::min(ny, _x + 6); ++x) {
    for (int y = _y; y < std::min(ny, _y + 16); ++y) {
      result[x + y * ny] = C[x - _x][y - _y];
    }
  }
}

void correlate(int ny, int nx, const float *data, float *result) {

  // copy the data to padded vector array
  // std::vector<double4_t> X(ncd * na, d4zero);
  std::vector<double> X(ny * nx, 0);

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
      X[x + y * nx] = (data[x + y * nx] - mean) / std;
    }
  }

  // normalize the rows
  for (int y = 0; y < ny; y++) {
    double sum = 0;
    for (int x = 0; x < nx; x++) {
      sum += X[x + y * nx] * X[x + y * nx];
    }
    double norm = sqrt(sum);
    for (int x = 0; x < nx; x++) {
      X[x + y * nx] /= norm;
    }
  }

//   // Number of padded columns
//   int const nc = (ny + nb - 1) / nb;

//   // Row-major vectorized padded matrix
//   std::vector<double4_t> Xt(nc * nb * nx, d4zero);

//   // Copy the data to the padded matrix
// #pragma omp parallel for collapse(2)
//   for (int y = 0; y < ny; y++) {
//     for (int x = 0; x < nx; x++) {
//       Xt[x / nb + y * nc][x % nb] = X[x + y * nx];
//     }
//   }

// compute XX^T
// C_ij = sum_k X_ik * X_jk
#pragma omp parallel for schedule(dynamic)
  for (int j = 0; j < ny; j += 16) {
    for (int i = j; i < ny; i += 6) {
      kernel(result, X.data(), i, j, nx, ny);
    }
  }
}

void correlate_vec_register(int ny, int nx, const float *data, float *result) {
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
