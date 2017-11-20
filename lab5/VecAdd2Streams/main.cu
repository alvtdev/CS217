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
    float *A_d0, *B_d0, *C_d0;
    float *A_d1, *B_d1, *C_d1;
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
    //TODO: modify to use cudaHostAlloc instead of malloc
    /*
    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );
    */
    cudaHostAlloc(&A_h, sizeof(float)*A_sz, cudaHostAllocDefault);
    cudaHostAlloc(&B_h, sizeof(float)*B_sz, cudaHostAllocDefault);
    cudaHostAlloc(&C_h, sizeof(float)*C_sz, cudaHostAllocDefault);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

		//INSERT CODE HERE
		//create streams first before allocating memory
		cudaStream_t stream0, stream1;
		cudaStreamCreate(&stream0); 
		cudaStreamCreate(&stream1); 
    //allocate memory for vectors A, B, and C on the device.
    //also error check for each malloc
    cudaError_t err_A =  cudaMalloc((void**)&A_d0, A_sz/2*sizeof(float));
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
   	err_A =  cudaMalloc((void**)&A_d1, A_sz/2*sizeof(float));
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    cudaError_t err_B = cudaMalloc((void**)&B_d0, B_sz/2*sizeof(float));
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_B = cudaMalloc((void**)&B_d1, B_sz/2*sizeof(float));
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    cudaError_t err_C = cudaMalloc((void**)&C_d0, C_sz/2*sizeof(float));
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_C = cudaMalloc((void**)&C_d1, C_sz/2*sizeof(float));
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
    //cudaMemcpy(A_d, A_h, A_sz*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(B_d, B_h, B_sz*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    /*** basic non-stream kernel call ***/
    ///basicVecAdd(A_d, B_d, C_d, A_h, B_h, C_h, VecSize); //In kernel.cu

    /***muli-stream code***/
    int segSize = VecSize/2+1;
    for (int i = 0; i < VecSize; i+= segSize*2) {
    	cudaMemcpyAsync(A_d0, A_h+i, segSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    	cudaMemcpyAsync(B_d0, B_h+i, segSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    	cudaMemcpyAsync(A_d1, A_h+i+segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
    	cudaMemcpyAsync(B_d1, B_h+i+segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream1);

    	VecAdd<<<segSize/256 + 1, 256, 0, stream0>>>(segSize, A_d0, B_d0, C_d0); 
    	VecAdd<<<segSize/256 + 1, 256, 0, stream1>>>(segSize, A_d1, B_d1, C_d1); 
			
			cudaMemcpyAsync(C_h+i, C_d0, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
			cudaMemcpyAsync(C_h+i+segSize, C_d0, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
		}

    cuda_ret = cudaDeviceSynchronize();
		if(cuda_ret != cudaSuccess) {
    	printf("CUDA failure %s:%d: '%s'\n", __FILE__,__LINE__,
    			cudaGetErrorString(cuda_ret)); 
			FATAL("Unable to launch/execute kernel");
		}
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    //copy C (result) back to host
    //cudaMemcpy(C_h, C_d, C_sz*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

		//TODO: modify to use cudaFreeHost
    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);

    //INSERT CODE HERE
    //free memory allocated in the device
    //also error check for each cudaFree
    err_A = cudaFree(A_d0);
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_A = cudaFree(A_d1);
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_B = cudaFree(B_d0);
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_B = cudaFree(B_d1);
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_C = cudaFree(C_d0);
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_C = cudaFree(C_d1);
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    return 0;

}
