//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define NUM_DIGITS 10

//
// Power function: had problems with powf
// 
__device__ int myPow(int base, int pow)
{
    int value = base;
    if (pow == 0)  value = 1;
    else
    {        
        for (int i = 0; i < pow- 1; i++)
        {
            value *= base;
        }
    }

    return value;
}

//
// Generate the histogram
//
__global__ void generateHistogram(unsigned int* const d_input,
                                  unsigned int* d_histo,
                                  int offset,
                                  int numEle)
{
    // Calculate the id
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numEle) return;

    // Calculate the histogram bin and atomically increase it
    unsigned int value = d_input[id] / myPow(10, offset);
    unsigned int histoIndex = value % 10;
    atomicAdd(&(d_histo[histoIndex]), 1);
}

//
// Prefix scan
//
__global__ void prefixScan(unsigned int* const d_values,
    unsigned int* d_scanOutput,
    unsigned int* d_scanOutputTemp,
    int offset,
    unsigned int checkNumber,
    int numEle)
{
    extern __shared__ int temp[];

    // Get Id
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int localId = threadIdx.x;
    temp[localId] = 0;

    if (id >= numEle) return;

    // Get the digit 
    unsigned int divisionFactor = myPow(10, offset);
    unsigned int testValue = (unsigned int)(d_values[id] / divisionFactor);
    testValue = testValue  % 10;

    //
    // If the digit is the number we are currently scanning set it to 1 so the sum scan works
    //
    if (testValue == checkNumber)
    {
        temp[localId] = 1;
    }
    __syncthreads();

    int tempOffset = 1;
    for (int counter = 0; counter < ceilf(log2f(blockDim.x)); counter++)
    {
        // Get the value of the offset if the Id is less then the offset store 0
        unsigned int tempA = 0;
        if (localId >= tempOffset)
        {
            tempA = temp[localId - tempOffset];
        }
        __syncthreads();

        // Add the value from the offset to the element
        temp[localId] += tempA;
        __syncthreads();

        // Double the offset
        tempOffset *= 2;
    }

    // Load total of blocks into the temp output
    if (localId == (blockDim.x - 1))  d_scanOutputTemp[blockIdx.x] = (temp[localId] > 0) ? temp[localId] : 0;
    
    // Load the block scan values into the output to be offset later
    if (testValue == checkNumber) d_scanOutput[id] = temp[localId];
}

//
// Scan all blocks from the prefix scan so they can be added to latter
//
__global__ void BlockScan(unsigned int* const d_values, int numEle)
{
    extern __shared__ int temp[];
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numEle) return;

    temp[id] = (id == 0) ? 0 : d_values[id - 1];
    __syncthreads();

    int tempOffset = 1;
    for (int counter = 0; counter < ceilf(log2f(blockDim.x)); counter++)
    {
        // Get the value of the offset if the Id is less then the offset store 0
        unsigned int tempA = 0;
        if (id >= tempOffset)
        {
            tempA = temp[id - tempOffset];
        }
        __syncthreads();

        // Add the value from the offset to the element
        temp[id] += tempA;
        __syncthreads();

        // Double the offset
        tempOffset *= 2;
    }

    d_values[id] = temp[id];
}

//
// Reduces all the blocks of the scan together
//
__global__ void OffsetScan(unsigned int* const d_values,
    unsigned int* d_scanOutput,
    unsigned int* d_scanOutputTemp,
    int offset,
    unsigned int checkNumber,
    int numEle)
{
    // Get global and local id
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numEle) return;

    int localId = threadIdx.x;    

    // Get the digit using offset
    unsigned int divisionFactor = myPow(10, offset);
    unsigned int testValue = (unsigned int)(d_values[id] / divisionFactor);
    testValue = testValue % 10;

    // Return if the digit is not the one we are scaning
    if (testValue != checkNumber) return;

    // Add the result of the block scans to all indexs
    d_scanOutput[id] += d_scanOutputTemp[blockIdx.x];

    // Subtract one to make everything 0 indexed
    if (d_scanOutput[id] != 0) d_scanOutput[id] -= 1;
}

// Scan histogram to get the offsets for the radix sort
__global__ void ScanHisto(unsigned int* const d_histo,                          
                          unsigned int* d_scanOutput,
                          const size_t numDigits)
{
    extern __shared__ int temp[];

    // Get id
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numDigits) return;

    // Load into shared memory
    temp[id] = (id == 0) ? 0 : d_histo[id - 1];
    __syncthreads();

    int offset = 1;
    for (int counter = 0; counter < ceilf(log2f(numDigits)); counter++)
    {
        // Get the value of the offset if the Id is less then the offset store 0
        unsigned int tempA = 0;
        if (id >= offset && id - offset >= 0)
        {
            tempA = temp[id - offset];
        }
        __syncthreads();

        // Add the value from the offset to the element
        temp[id] += tempA;
        __syncthreads();

        // Double the offset
        offset *= 2;
    }

    // Load from shared mem
    if (id < numDigits) d_scanOutput[id] = temp[id];
}

//
// radix sort step
//
__global__ void reSortValues(unsigned int* const d_values,
    unsigned int* const d_positions,
    unsigned int* const d_scanedHisto,
    unsigned int* d_scanOutput,
    unsigned int* const d_outputValues,
    unsigned int* const d_outputPositions,
    int offset,
    int numEle)
{
    // Get Id
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= numEle) return;

    // Get histogram index
    int divisionFactor = myPow(10, offset);
    unsigned int histoIndex = (unsigned int)(d_values[id] / divisionFactor) % 10;

    // Calculate the output Index by adding the offset from the histrogram and the offset from the 
    // output scan
    int outputIndex = d_scanedHisto[histoIndex] + d_scanOutput[id];        

    // Move the value and position 
    d_outputValues[outputIndex] = d_values[id];
    d_outputPositions[outputIndex] = d_positions[id];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    // Set up Block and thread sizes
    unsigned int numToUse = numElems;
    int numThreads = 1024;
    int numBlocks = ceil((float)numToUse / (float)numThreads);

    //
    // Set up cuda memory
    //
    unsigned int* d_histoScanOutput;
    unsigned int* d_histo;
    unsigned int* d_prefixScanOutput;
    unsigned int* d_prefixScanTemp;

    checkCudaErrors(cudaMalloc((void**)&d_histoScanOutput, sizeof(unsigned int)* NUM_DIGITS));
    checkCudaErrors(cudaMalloc((void**)&d_histo, sizeof(unsigned int)* NUM_DIGITS));
    checkCudaErrors(cudaMalloc((void**)&d_prefixScanOutput, sizeof(unsigned int)* numToUse));
    checkCudaErrors(cudaMalloc((void**)&d_prefixScanTemp, sizeof(unsigned int)* numToUse));

    //
    // Loop through all possible digits
    //
    for (int offset = 0; offset < 14; offset++)
    {
        //
        // Clear memory from last run
        //
        checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int)* NUM_DIGITS));
        checkCudaErrors(cudaMemset(d_histoScanOutput, 0, sizeof(unsigned int)* NUM_DIGITS));
        checkCudaErrors(cudaMemset(d_prefixScanTemp, 0, sizeof(unsigned int)* NUM_DIGITS));
        checkCudaErrors(cudaMemset(d_prefixScanOutput, 0, sizeof(unsigned int)* numToUse));

        // Generate Histogram
        generateHistogram <<<numBlocks, numThreads>>>(d_inputVals,
                                                        d_histo,
                                                        offset,
                                                        numToUse);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaGetLastError());
   
        //
        // Find Relative offsets for each digit individually 
        //
        for (int i = 0; i < NUM_DIGITS; i++)
        {
            checkCudaErrors(cudaMemset(d_prefixScanTemp, 0, sizeof(unsigned int)* NUM_DIGITS));

            // Find relative offsets
            prefixScan << <numBlocks, numThreads, sizeof(int) * numThreads>> >(d_inputVals,
                d_prefixScanOutput,
                d_prefixScanTemp,
                offset,
                i,
                numToUse);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            BlockScan << <1, numBlocks, numBlocks * sizeof(int) >> >(d_prefixScanTemp, numBlocks);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            OffsetScan << <numBlocks, numThreads>> >(d_inputVals,
                                                     d_prefixScanOutput,
                                                     d_prefixScanTemp,
                                                     offset,
                                                     i,
                                                     numToUse);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        }
        
        // Scan the histogram to determine offsets
        ScanHisto << <1, NUM_DIGITS, sizeof(unsigned int)*  NUM_DIGITS >> >(d_histo,
                                                                       d_histoScanOutput,
                                                                       NUM_DIGITS);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // Sort the histogram
        reSortValues << <numBlocks, numThreads >> >(d_inputVals,
                                                    d_inputPos,
                                                    d_histoScanOutput,
                                                    d_prefixScanOutput,
                                                    d_outputVals,
                                                    d_outputPos,
                                                    offset,
                                                    numToUse);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, sizeof(unsigned int)* numToUse, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, sizeof(unsigned int)* numToUse, cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_histo);
    cudaFree(d_histoScanOutput);
    cudaFree(d_prefixScanOutput);
}
