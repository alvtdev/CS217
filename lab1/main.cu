/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize;
   
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 1000000;

      } else if (argc == 2) {
      VecSize = atoi(argv[1]);   
      
      
      }
  
      else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

		//INSERT CODE HERE
    //allocate memory for vectors A, B, and C on the device.
    //also error check for each malloc
    cudaError_t err_A =  cudaMalloc((void**)&A_d, A_sz*sizeof(float));
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    cudaError_t err_B = cudaMalloc((void**)&B_d, B_sz*sizeof(float));
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    cudaError_t err_C = cudaMalloc((void**)&C_d, C_sz*sizeof(float));
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    //only copy A and B from host to device
    //vector sum result will be in C, which is copied from device to host
    cudaMemcpy(A_d, A_h, A_sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_sz*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicVecAdd(A_d, B_d, C_d, VecSize); //In kernel.cu

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    //copy C (result) back to host
    cudaMemcpy(C_h, C_d, C_sz*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    //free memory allocated in the device
    //also error check for each cudaFree
    err_A = cudaFree(A_d);
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_B = cudaFree(B_d);
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_C = cudaFree(C_d);
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    return 0;

}
