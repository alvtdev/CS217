/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
    

__global__ void VecAdd(int n, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
  	// get thread id
  	int tid = blockIdx.x * blockDim.x + threadIdx.x;

  	// if thread id is within bounds, add
  	if (tid < n) C[tid] = A[tid] + B[tid];

}


void basicVecAdd( float *A,  float *B, float *C, int n)
{

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 256; 

    //INSERT CODE HERE
    //call __global VecAdd__
    //calculate number of blocks needed -- ceil(n/256.0)
    //256.0 = BLOCK_SIZE, but in floating point.
    VecAdd<<<ceil(n/256.0), BLOCK_SIZE>>>(n, A, B, C);

}

