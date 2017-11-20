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
    float *A_d2, *B_d2, *C_d2;
    float *A_d3, *B_d3, *C_d3;
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
		cudaStream_t stream0, stream1, stream2, stream3;
		cudaStreamCreate(&stream0); 
		cudaStreamCreate(&stream1); 
		cudaStreamCreate(&stream2); 
		cudaStreamCreate(&stream3); 
    //allocate memory for vectors A, B, and C on the device.
    //also error check for each malloc

    //TODO:ALLOCATE MEMORY FOR A0, A1, A2, A3
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
   	err_A =  cudaMalloc((void**)&A_d2, A_sz/2*sizeof(float));
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
   	err_A =  cudaMalloc((void**)&A_d3, A_sz/2*sizeof(float));
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    //TODO: ALLOCATE MEMORY FOR B0, B1, B2, B3
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
    err_B = cudaMalloc((void**)&B_d2, B_sz/2*sizeof(float));
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_B = cudaMalloc((void**)&B_d3, B_sz/2*sizeof(float));
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    //TODO: ALLOCATE MEMORY FOR C0, C1, C2, C3
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
    err_C = cudaMalloc((void**)&C_d2, C_sz/2*sizeof(float));
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_C = cudaMalloc((void**)&C_d3, C_sz/2*sizeof(float));
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
    //TODO: MODIFY TO USE 4 STREAMS INSTEAD OF 2
    int segSize = VecSize/4+1;
    for (int i = 0; i < VecSize; i+= segSize*4) {
    	cudaMemcpyAsync(A_d0, A_h+i, segSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    	cudaMemcpyAsync(B_d0, B_h+i, segSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
    	cudaMemcpyAsync(A_d1, A_h+i+segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
    	cudaMemcpyAsync(B_d1, B_h+i+segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream1);
    	cudaMemcpyAsync(A_d2, A_h+i+2*segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream2);
    	cudaMemcpyAsync(B_d2, B_h+i+2*segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream2);
    	cudaMemcpyAsync(A_d3, A_h+i+3*segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream3);
    	cudaMemcpyAsync(B_d3, B_h+i+3*segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream3);

    	VecAdd<<<segSize/256 + 1, 256, 0, stream0>>>(segSize, A_d0, B_d0, C_d0); 
    	VecAdd<<<segSize/256 + 1, 256, 0, stream1>>>(segSize, A_d1, B_d1, C_d1); 
    	VecAdd<<<segSize/256 + 1, 256, 0, stream2>>>(segSize, A_d2, B_d2, C_d2); 
    	VecAdd<<<segSize/256 + 1, 256, 0, stream3>>>(segSize, A_d3, B_d3, C_d3); 
			
			cudaMemcpyAsync(C_h+i, C_d0, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
			cudaMemcpyAsync(C_h+i+segSize, C_d1, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
			cudaMemcpyAsync(C_h+i+2*segSize, C_d2, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream2);
			cudaMemcpyAsync(C_h+i+3*segSize, C_d3, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream3);
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

    //TODO: FREE A0, A1, A2, A3
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
    err_A = cudaFree(A_d2);
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_A = cudaFree(A_d3);
    if (err_A != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    //TODO: B0, B1, B2, B3
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
    err_B = cudaFree(B_d2);
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_B = cudaFree(B_d3);
    if (err_B != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_B), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    //TODO: C0, C1, C2, C3
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
    err_C = cudaFree(C_d2);
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    err_C = cudaFree(C_d3);
    if (err_C != cudaSuccess) {
    	printf("%s in %s at line %d\n", cudaGetErrorString(err_C), __FILE__, __LINE__);
    	exit(EXIT_FAILURE);
    }
    return 0;

}
