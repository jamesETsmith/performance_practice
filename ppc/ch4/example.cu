__device__ void foo(int i, int j) {}

__global__ void mykernel() {
  int i = blockIdx.x;
  int j = threadIdx.x;
  foo(i, j);
}

int main() {
  mykernel<<<100, 128>>>();
  cudaDeviceSynchronize();
}