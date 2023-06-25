#include <stdio.h>  /* For printf() function */
#include <stdlib.h> /* For status marcos */

/**********************************************/

void helloFromCPU(void) { /* This function runs on the host */
  printf("Hello World from CPU!\n");
}

__global__ void helloFromGPU() { /* This kernel is launched on the device */
  printf("Hello World from GPU!\n");
}

/**********************************************/

int main(int argc, char **argv) {
  helloFromCPU(); /* Calling from host */

  helloFromGPU<<<1, 1>>>(); /* Launching from the host */

  cudaDeviceReset(); /* House-keeping on the device */

  return (EXIT_SUCCESS);
}