#ifndef HELPER_H
#define HELPER_H

#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdlib>
#include <iostream>

void fill_matrix(float** f, int size) {
  *f = (float*)malloc(size * sizeof(float));
  srand(0);

  for (int i = 0; i < size; i++) {
    (*f)[i] = rand();
  }
}

static inline void check(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << context << ": " << cudaGetErrorString(err)
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) { return (a + b - 1) / b; }

static inline int roundup(int a, int b) { return divup(a, b) * b; }

__global__ void mykernel(float* r, const float* d, int n);

void step(float* r, const float* d, int n) {
  // Allocate memory & copy data to GPU
  float* dGPU = NULL;
  CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
  float* rGPU = NULL;
  CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
  CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel
  dim3 dimBlock(16, 16);
  dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
  mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
  CHECK(cudaGetLastError());

  // Copy data back to CPU & release memory
  CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dGPU));
  CHECK(cudaFree(rGPU));
}

#endif