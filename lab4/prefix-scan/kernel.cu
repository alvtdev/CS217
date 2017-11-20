/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE
__global__ void prefix_scan(float *out, float *in, float* aux, unsigned in_size) {

	__shared__ float XY[2*BLOCK_SIZE];
	//extern __shared__ float XY[];
	//indexing in shared memory
	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	int start = 2*blockIdx.x*blockDim.x;
	int tid = threadIdx.x;
	/*
	if (i < in_size) {
		XY[threadIdx.x] = in[i];
	}
	*/
	/*
	XY[2*tid] = in[2*tid];
	XY[2*tid + 1] = in[2*tid+1];
	*/
	if (start + tid < in_size) {
		XY[tid] = in[start + tid];
	} 
	else {
		XY[tid] = 0;
	}
	if (start + blockDim.x + tid < in_size) {
		XY[blockDim.x + tid] = in[start + blockDim.x + tid];
	}
	else {
		XY[blockDim.x + tid] = 0;
	}
	__syncthreads();

	//reduction
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2*BLOCK_SIZE) {
			XY[index] += XY[index - stride]; 
		} 
		__syncthreads();
	}

	/*
	if (tid == 0) {
		XY[2*BLOCK_SIZE - 1] = 0; 
	}
	*/


	//post-reduction
	for (unsigned int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE) {
			XY[index + stride] += XY[index];
		} 
	} 
	__syncthreads();

	//copy from shared to global mem
	/*
	if (i < in_size) {
		out[i] = XY[threadIdx.x];
	}
	*/
	/*
	out[2*tid] = XY[2*tid];
	out[2*tid + 1] = XY[2*tid+1];
	*/
	if (start + tid < in_size) {
		out[start + tid] = XY[tid];
	}
	if (start + blockDim.x + tid < in_size) {
		out[start + blockDim.x + tid] = XY[blockDim.x + tid] ;
	}
	if (tid == 0 && aux != NULL) {
		aux[blockIdx.x] = XY[2 * BLOCK_SIZE - 1];
	}

	return;
}

//kernel to add scanned auxiliary array to output array
__global__ void addAuxArray(float *in, float *aux, unsigned in_size) {
	int tid = threadIdx.x;
	int start = 2 * blockIdx.x * blockDim.x;
	if (blockIdx.x > 0) {
		if (start + tid < in_size) {
			in[start+tid] += aux[blockIdx.x-1];
		}
		if (start + blockDim.x + tid < in_size) {
			in[start + blockDim.x + tid] += aux[blockIdx.x-1];
		}
	} 
	return;
} 

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
    // INSERT CODE HERE
  //declare auxiliary arrays
  float *dAuxArr;
  float *dAuxScanned;
  //allocate memory for auxiliary arrays
  cudaMalloc(&dAuxArr, 2*BLOCK_SIZE*sizeof(float));
  cudaMalloc(&dAuxScanned, 2*BLOCK_SIZE*sizeof(float));
  //determine grid and block sizes
  dim3 dimGrid(in_size/(2*BLOCK_SIZE) + 1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  
  //scan and store block sums in auxiliary array
  //also works as prefix sum for inputs of size < 2*BLOCK_SIZE
  prefix_scan<<<dimGrid, dimBlock>>>(out, in, dAuxArr, in_size);

  //THE CODE AFTER THIS POINT IS AN ATTEMPT AT MAKING PREFIX SCAN WORK FOR 
  //INPUTS OF SIZE > 2 * BLOCK_SIZE

	cudaDeviceSynchronize();
	//scan the block sums
	prefix_scan<<<dim3(1, 1, 1), dimBlock>>>(dAuxScanned, dAuxArr, NULL, 2*BLOCK_SIZE);
	cudaDeviceSynchronize();

	//add scanned block sum to output
	addAuxArray<<<dimGrid, dimBlock>>>(out, dAuxScanned, in_size);
	cudaDeviceSynchronize();

	cudaFree(dAuxArr);
	cudaFree(dAuxScanned);

}

