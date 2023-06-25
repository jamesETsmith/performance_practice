#include "helper.h"

__global__ void mykernel(float* r, const float* d, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= n || j >= n) return;
  float v = HUGE_VALF;
  for (int k = 0; k < n; ++k) {
    float x = d[n * i + k];
    float y = d[n * k + j];
    float z = x + y;
    v = min(v, z);
  }
  r[n * i + j] = v;
}

int main() {
  int n = 6300;
  float* d;
  fill_matrix(&d, n * n);
  float* r = (float*)malloc(n * n * sizeof(float));
  step(r, d, n);
  return 0;
}