/* Udacity HW5
Histogramming for Speed

The goal of this assignment is compute a histogram
as fast as possible.  We have simplified the problem as much as
possible to allow you to focus solely on the histogramming algorithm.

The input values that you need to histogram are already the exact
bins that need to be updated.  This is unlike in HW3 where you needed
to compute the range of the data and then do:
bin = (val - valMin) / valRange to determine the bin.

Here the bin is just:
bin = val

so the serial histogram calculation looks like:
for (i = 0; i < numElems; ++i)
histo[val[i]]++;

That's it!  Your job is to make it run as fast as possible!

The values are normally distributed - you may take
advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"
#include <thrust/sort.h>
#include <iostream>

__global__
void CoarseBins(const unsigned int* const vals, //INPUT
unsigned int* outHisto,
int numVals,
int numCoarseBins,
int numBinsPerCoarseBins,
int numBins)
{
    __shared__ int lowestId;
    int id = threadIdx.x;
    extern __shared__ unsigned int localhisto[];

    if (threadIdx.x == 0) lowestId = numBinsPerCoarseBins * blockIdx.x;
    if (threadIdx.x < numBinsPerCoarseBins) localhisto[threadIdx.x] = 0;
    __syncthreads();

    while (id < numVals)
    {
        if ((vals[id] / numBinsPerCoarseBins) == blockIdx.x) atomicAdd(&(localhisto[vals[id] - lowestId]), 1);
        id += blockDim.x;
    }

    __syncthreads();

    if (threadIdx.x < numBinsPerCoarseBins)
    {
        int offset = numBinsPerCoarseBins * blockIdx.x + threadIdx.x;
        if (offset < numBins) outHisto[offset] = localhisto[threadIdx.x];
    }
}

void computeHistogram(const unsigned int* const d_vals,
    unsigned int* const d_histo,
    const unsigned int numBins,
    const unsigned int numElems)
{
    int numBinsPerCoarse = 32;
    int numCoarseBins = ceil(numBins / numBinsPerCoarse);
    int numThreads = 1024;

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    CoarseBins << <numCoarseBins, numThreads, sizeof(int)* numBinsPerCoarse >> >(d_vals,
        d_histo,
        numElems,
        numCoarseBins,
        numBinsPerCoarse,
        numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
