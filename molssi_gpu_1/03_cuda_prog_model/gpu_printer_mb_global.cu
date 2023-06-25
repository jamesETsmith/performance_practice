#include <stdio.h>
#include <stdlib.h>

#define N 8

void cpuPrinter(int nlim) {
  for (int idx = 0; idx < nlim; idx++) printf("CPU Prints Idx: %d\n", idx);

  printf("\n");
}

__global__ void gpuPrinter(void) {
  int idx =
      threadIdx.x +
      blockIdx.x * blockDim.x; /* The local thread index (threadIdx.x) in the
                                  right hand side should be shifted by an offset
                                  value (blockIdx.x * blockDim.x) to compensate
                                  translate it to a global index */
  printf("GPU Prints Idx: %d\n",
         idx); /* Write the kernel for individual threads */
}

int main(int argc, char **argv) {
  cpuPrinter(N);

  gpuPrinter<<<2, N / 2>>>(); /*  Organizing eight threads in two blocks with
                                 four threads */

  cudaDeviceSynchronize();

  return (EXIT_SUCCESS);
}