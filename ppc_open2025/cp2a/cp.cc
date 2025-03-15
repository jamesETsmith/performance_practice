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

void correlate(int ny, int nx, const float *data, float *result) {
  // All arithmetic must be in double precision per the problem instructions
  // online

  // pad X to a block size of 4
  int constexpr nb = 8;
  int na = (nx + nb - 1) / nb;
  int nx_padded = na * nb;

  std::vector<double> X(ny * nx_padded, 0);

  // standardize the rows (set the mean to 0 and std to 1)
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
      X[x + y * nx_padded] = (data[x + y * nx] - mean) / std;
    }
  }

  // normalize the rows
  for (int y = 0; y < ny; y++) {
    double sum = 0;
    for (int x = 0; x < nx_padded; x++) {
      sum += X[x + y * nx_padded] * X[x + y * nx_padded];
    }
    double norm = sqrt(sum);
    for (int x = 0; x < nx_padded; x++) {
      X[x + y * nx_padded] /= norm;
    }
  }

  // compute XX^T
  for (int j = 0; j < ny; ++j) {
    for (int i = j; i < ny; ++i) {

      // Break up the inner product into a sum of nb smaller inner products
      std::array<double, nb> accum = {0};

      // Iterate over blocks of nb size
      for (int xa = 0; xa < na; ++xa) {
        // iterate over the block
        for (int xb = 0; xb < nb; ++xb) {
          int x = xa * nb + xb;
          accum[xb] += X[x + i * nx_padded] * X[x + j * nx_padded];
        }
      }

      double res = std::reduce(accum.begin(), accum.end());
      result[i + j * ny] = res;
    }
  }
}
