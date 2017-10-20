/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

  	__shared__ float ds_A[TILE_SIZE][TILE_SIZE];
  	__shared__ float ds_B[TILE_SIZE][TILE_SIZE];

  	//declare and initialize ids
  	int bx = blockIdx.x;
  	int by = blockIdx.y;
  	int tx = threadIdx.x;
  	int ty = threadIdx.y;

		//calculate row/column
		int row = blockDim.y * by + ty; 
		int col = blockDim.x * bx + tx;
		
		//declare P value variable for matrix sum
		float pValue = 0.0;

		//compute the P element by looping over M and N matrices
		for (int p = 0; p < ((k-1)/TILE_SIZE+1) ; p++) {
			//load A and B tiles into shared memory
			//loads 0 into memory if out of bounds
			if(row < m && (p*TILE_SIZE + tx) < k) {
				ds_A[ty][tx] = A[row*k + p*TILE_SIZE + tx];
			}
			else {
				ds_A[ty][tx] = 0.0;
			}
			if (col < n && (p*TILE_SIZE + ty) < k) {
				//ds_B[ty][tx] = B[(p*TILE_SIZE*+ty)*n + col];
			}
			else {
				ds_B[ty][tx] = 0.0;
			}
			__syncthreads();

			//compute p value 
			if (row < m && col < n) {
				for (int j = 0; j < TILE_SIZE; j++) {
					pValue += ds_A[ty][j] * ds_B[j][tx];
				}
			}
			__syncthreads();
		}

		//assign pValues to C if within bounds
		if(row < m && col < n) {
			C[row*n+col] = pValue;
		}

}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
		//printf("INIT THREADS AND KERNEL GRIDS...\n"); fflush(stdout);

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
		dim3 dimGrid(((n-1)/TILE_SIZE)+1, ((m-1)/TILE_SIZE)+1);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);



    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);

}


