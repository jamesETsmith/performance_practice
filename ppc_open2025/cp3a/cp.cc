/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>
//
// User defined constants
//
int constexpr nv = 4;

int constexpr nr = 4; // right now we make this same size as our vector
int constexpr nc = 4;

int constexpr jblock = nc * 10;
int constexpr iblock = nr * 32;
int constexpr kblock = 32;

static_assert(iblock % 4 == 0);
static_assert(jblock % 4 == 0);

//
// User defined types
//
using double4_t = double __attribute__((vector_size(nv * sizeof(double))));

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
template <int nr, int nc, int kblock>
void kernel(double *result, double const *X, int _x, int _y, int const k0,
            int const nx, int const ny) {

  // C[nc][nr] make it column-major
  __m256d C[nr]{};

  int const kend = std::min(k0 + kblock, nx);

  // create A[kblock][nr] so it's column-major
  double A[kblock][nr]{};

  // load A
  for (int x = 0; x < std::min(nr, ny - _x); ++x) {
    for (int k = k0; k < kend; ++k) {
      A[k - k0][x] = X[k + (_x + x) * nx];
    }
  }

  // create B[kblock][nc]
  double B[kblock][nc]{};

  // load B
  for (int k = k0; k < kend; ++k) {
    for (int y = 0; y < std::min(nc, ny - _y); ++y) {
      B[k - k0][y] = X[k + (_y + y) * nx];
    }
  }

  // compute C[nr][nc] = A[nr][kblock] * B[kblock][nc]
  // using rank-1 updates
  __m256d a_col;
  __m256d b_col;
  // __m256d c_col;

  for (int k = 0; k < kblock; ++k) {
    a_col = _mm256_loadu_pd(&A[k][0]);
    // c_col = _mm256_setzero_pd();
    for (int j = 0; j < nc; ++j) {
      b_col = _mm256_broadcast_sd(&B[k][j]);
      C[j] = _mm256_fmadd_pd(a_col, b_col, C[j]);
    }
    // _mm256_store_pd(&C[j][0], c_col);
  }

  // store C
  // for (int j = 0; j < std::min(nc, ny - _y); ++j) {
  //   for (int i = 0; i < std::min(nr, ny - _x); ++i) {
  //     result[(_x + i) + (_y + j) * ny] += C[j][i];
  //   }
  //   // _mm256_storeu_pd(&result[_x + (_y + j) * ny], C[j]);
  // }
  double temp[4] = {};
  for (int j = 0; j < std::min(nc, ny - _y); ++j) {
    _mm256_storeu_pd(temp, C[j]);
    for (int i = 0; i < std::min(nr, ny - _x); ++i) {
      result[(_x + i) + (_y + j) * ny] += temp[i];
    }
  }

  // for (int x = _x; x < std::min(ny, _x + nr); ++x) {

  //   for (int y = _y; y < std::min(ny, _y + nc); ++y) {

  //     // Do a chunk of the row
  //     for (int k = k0; k < kend; ++k) {
  //       C[x - _x][y - _y] += X[k + x * nx] * X[k + y * nx];
  //     }
  //   }
  // }

  // // copy C to the results
  // for (int x = _x; x < std::min(ny, _x + nr); ++x) {
  //   for (int y = _y; y < std::min(ny, _y + nc); ++y) {
  //     result[x + y * ny] += C[x - _x][y - _y];
  //   }
  // }
}

// void kernel_v2(double *result, double4_t *X1, double4_t *X2, int x1, int y1,
//                int x2, int y2, int k0, int kend) {}

void correlate(int ny, int nx, const float *data, float *result) {

  // copy the data to padded vector array
  // std::vector<double4_t> X(ncd * na, d4zero);
  std::vector<double> X(ny * nx, 0);

  // standardize the rows (set the mean to 0 and std to 1)
#pragma omp parallel for
  for (int y = 0; y < ny; y++) {
    double mean = 0;
#pragma omp simd reduction(+ : mean)
    for (int x = 0; x < nx; x++) {
      mean += data[x + y * nx];
    }
    mean /= nx;
    double std = 0;
#pragma omp simd reduction(+ : std)
    for (int x = 0; x < nx; x++) {
      std += (data[x + y * nx] - mean) * (data[x + y * nx] - mean);
    }
    // Use the population standard deviation because we are using the entire
    // dataset
    std = sqrt(std / nx);
#pragma omp simd
    for (int x = 0; x < nx; x++) {
      X[x + y * nx] = (data[x + y * nx] - mean) / std;
    }
  }

  // normalize the rows
#pragma omp parallel for
  for (int y = 0; y < ny; y++) {
    double sum = 0;
#pragma omp simd reduction(+ : sum)
    for (int x = 0; x < nx; x++) {
      sum += X[x + y * nx] * X[x + y * nx];
    }
    double norm = sqrt(sum);
#pragma omp simd
    for (int x = 0; x < nx; x++) {
      X[x + y * nx] /= norm;
    }
  }

  // compute XX^T
  // C_ij = sum_k X_ik * X_jk
  // #pragma omp parallel for schedule(dynamic)

  // zero out the result
  // memset(result, 0, ny * ny * sizeof(float));
  std::vector<double> result_d(ny * ny, 0);

  // #pragma omp parallel for schedule(dynamic)
  //   for (int j = 0; j < ny; j += 16) {
  //     for (int i = j; i < ny; i += 6) {
  //       kernel(result_d.data(), X.data(), i, j, 0, nx, nx, ny);
  //     }
  //   }

  // block loop over j

  // double4_t X1[iblock / 4][kblock]{}, X2[jblock / 4][kblock]{};

// break up the X_jk into blocks of rows
#pragma omp parallel for schedule(dynamic)
  for (int j = 0; j < ny; j += jblock) {

    // break X_ik_block into blocks of rows
    for (int i = j; i < ny; i += iblock) {

      // break up X_jk_block into blocks of columns
      for (int k = 0; k < nx; k += kblock) {

        // move the C_ij_block throughout the blocks
        for (int jj = j; jj < std::min(ny, j + jblock); jj += nc) {
          for (int ii = i; ii < std::min(ny, i + iblock); ii += nr) {
            kernel<nr, nc, kblock>(result_d.data(), X.data(), ii, jj, k, nx,
                                   ny);
            // kernel(result, X.data(), ii, jj, k, kend, nx, ny);
          }
        }
      }
    }
  }

#pragma omp parallel for simd
  for (int i = 0; i < ny * ny; i++) {
    result[i] = result_d[i];
  }
}

// template <int iblock, int kblock>
// void pack_X1(double4_t *X1, double const *X, int const _x, int const _y,
//              int const k0, int const kend, int const nx, int const ny) {
//   for (int i = 0; i < iblock / 4; i += 4) {
//     for (int k = 0; k < kblock; k++) {
//       for (int ii = 0; ii < 4; ii++) {
//         X1[i][k][ii] = X[k + (_x + ii) * nx];
//       }
//     }
//   }
// }

// void correlate_vec_register(int ny, int nx, const float *data, float
// *result)
// {
//   // Calculate the number of vectors we'll need for each row
//   int na = (nx + nb - 1) / nb;

//   // number of blocks we'll need for each column
//   int nc = (ny + nd - 1) / nd;
//   // Number of rows after padding
//   int ncd = nc * nd;

//   // copy the data to padded vector array
//   std::vector<double4_t> X(ncd * na, d4zero);

// #pragma omp parallel
//   {

// #pragma omp for collapse(2)
//     for (int y = 0; y < ny; y++) {
//       for (int x = 0; x < nx; x++) {
//         X[x / nb + y * na][x % nb] = data[x + y * nx];
//       }
//     }

//     // standardize the rows (set the mean to 0 and std to 1)

// #pragma omp for
//     for (int y = 0; y < ny; y++) {
//       // mean
//       double4_t mean_vec = d4zero;
//       for (int xa = 0; xa < na; xa++) {
//         mean_vec += X[xa + y * na];
//       }
//       // reduce and broadcast results
//       double mean = reduce_d4t(mean_vec) / nx;
//       for (int xb = 0; xb < nb; xb++) {
//         mean_vec[xb] = mean;
//       }

//       // std
//       double4_t std_vec = d4zero;
//       for (int xa = 0; xa < na; xa++) {
//         double4_t tmp = X[xa + y * na] - mean_vec;
//         std_vec += tmp * tmp;
//       }
//       // reduce and broadcast results
//       double std = sqrt(reduce_d4t(std_vec) / nx);
//       for (int xb = 0; xb < nb; xb++) {
//         std_vec[xb] = std;
//       }

//       // scale and calculate norm
//       double4_t norm_vec = d4zero;
//       for (int xa = 0; xa < ((nx % nb == 0) ? na : na - 1); xa++) {
//         int idx = xa + y * na;
//         X[idx] = (X[idx] - mean_vec) / std_vec;
//         norm_vec += X[idx] * X[idx];
//       }

//       // for the tail we only populate the non-padded elements
//       for (int xb = 0; xb < nx % nb; xb++) {
//         int idx = na - 1 + y * na;
//         X[idx][xb] = (X[idx][xb] - mean_vec[xb]) / std_vec[xb];
//         norm_vec[xb] += X[idx][xb] * X[idx][xb];
//       }

//       double norm = reduce_sqrt(norm_vec);
//       for (int xb = 0; xb < nb; xb++) {
//         norm_vec[xb] = norm;
//       }

//       for (int xa = 0; xa < na; ++xa) {
//         X[xa + y * na] /= norm_vec;
//       }
//     }

//     // compute XX^T
// #pragma omp for schedule(dynamic)
//     for (int jc = 0; jc < nc; ++jc) {
//       for (int ic = jc; ic < nc; ++ic) {

//         std::array<std::array<double4_t, nd>, nd> vv = {d4zero};
//         std::array<double4_t, nd> xi;
//         std::array<double4_t, nd> yi;

//         for (int ka = 0; ka < na; ++ka) {
//           // indices are a mess here
//           // (unpacked row idx) * na + vector idx

//           for (int i = 0; i < nd; ++i) {
//             xi[i] = X[(jc * nd + i) * na + ka];
//             yi[i] = X[(ic * nd + i) * na + ka];
//           }

//           for (int i = 0; i < nd; ++i) {
//             for (int j = 0; j < nd; ++j) {
//               vv[i][j] += xi[i] * yi[j];
//             }
//           }
//         }

//         // reduce vv and store in result
//         for (int id = 0; id < nd; ++id) {
//           for (int jd = 0; jd < nd; ++jd) {
//             int i = ic * nd + id;
//             int j = jc * nd + jd;

//             if (i < ny && j < ny) {
//               result[i + j * ny] = reduce_d4t(vv[jd][id]);
//             }
//           }
//         }
//       }
//     }
//   }
// }
