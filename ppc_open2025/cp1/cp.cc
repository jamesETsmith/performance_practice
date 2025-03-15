/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <vector>
void correlate(int ny, int nx, const float *data, float *result) {
  // standardize the rows (set the mean to 0 and std to 1)
  std::vector<double> X(ny * nx, 0);

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

  // compute XX^T
  for (int j = 0; j < ny; ++j) {
    for (int i = j; i < ny; ++i) {
      double res = 0;
      for (int x = 0; x < nx; ++x) {
        res += X[x + i * nx] * X[x + j * nx];
      }
      result[i + j * ny] = res;
    }
  }

  // compute the correlation between the rows
}
