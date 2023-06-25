#include "helper.h"

/*
- There is a very simple solution: just swap the role of the indexes i and j
- Now on the line float x = d[...] the memory addresses that the warp accesses
are good; we are accessing only two distinct elements:
- And on the line float y = d[...] the memory addresses that the warp accesses
are also good; we are accessing one continuous part of the memory:
*/
__global__ void mykernel(float* r, const float* d, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= n || j >= n) return;
  float v = HUGE_VALF;
  for (int k = 0; k < n; ++k) {
    float x = d[n * j + k];
    float y = d[n * k + i];
    float z = x + y;
    v = min(v, z);
  }
  r[n * j + i] = v;
}
int main() {
  int n = 6300;
  float* d;
  fill_matrix(&d, n * n);
  float* r = (float*)malloc(n * n * sizeof(float));
  step(r, d, n);
  return 0;
}