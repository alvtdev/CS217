/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

//extern __shared__ unsigned int Hist[];

__global__ void HistogramKernel(unsigned int *buffer, unsigned int numBins, 
			unsigned int* histo, unsigned int size) {


	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	//create and initialized shared mem 
	extern __shared__ unsigned int histo_private[];
	for(int j = threadIdx.x; j < numBins; j += blockDim.x) {
		histo_private[j] = 0;
	}
	__syncthreads();

	//fill shared mem histogram
	while (i < size) {
		atomicAdd(&(histo_private[(buffer[i])]), 1);
	  i += stride;
	}

	__syncthreads();

	//create final histogram using private histogram
	for(int j = threadIdx.x; j < numBins; j += blockDim.x) {
		atomicAdd(&(histo[j]), histo_private[j]);
	}
	__syncthreads();

	//naive implementation below:
	/*
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (i < size) {
		unsigned int num = buffer[i];
		atomicAdd(&histo[num], 1);
		i += stride;
	}
	*/


}





/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/


void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
  	
  //determine block/grid
  
  const unsigned int BLOCK_SIZE = 1024;

  HistogramKernel<<<ceil(num_elements/(float)BLOCK_SIZE), BLOCK_SIZE, 
  		sizeof(unsigned int)*num_bins>>>(input, num_bins, bins, num_elements);

}


