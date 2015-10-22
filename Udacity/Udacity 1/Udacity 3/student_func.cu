/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Definition Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/

#include "utils.h"
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#define MAX 1
#define MIN 0

__global__ void findMaxMin(const float* const d_logLuminance,
	const size_t numRows,
	const size_t numCols,
	float* out,
	int MinMax,
	int size)
{
	extern __shared__ float Max_Min[];

	// Get global x and y coordinates 
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	// Get id local to the bloack
	int localId = threadIdx.x + threadIdx.y * blockDim.x;

	// Calculate the pixel id
	int id = idx + idy * numCols;

	// If the id is out side of the range ignore it
	if (size <= id) return;

	// Load first compare into shared memory
	int offset = (blockDim.x * blockDim.y) / 2;
	if (localId < offset)
	{
		Max_Min[localId] = (MinMax == 1) ? max(d_logLuminance[id], d_logLuminance[id + offset]) : fminf(d_logLuminance[id], d_logLuminance[id + offset]);
	}
	__syncthreads();


	// Finish Comparing
	for (offset /= 2; offset > 0; offset /= 2)
	{
		if (localId < offset)
		{
			Max_Min[localId] = (MinMax == 1) ? max(Max_Min[localId], Max_Min[localId + offset]) : fminf(Max_Min[localId], Max_Min[localId + offset]);
		}

		__syncthreads();
	}

	// Load Result based on the block Id
	if (localId == 0) out[blockIdx.x + blockIdx.y * gridDim.x] = Max_Min[0];
}

__global__ void generateHistogram(const float* const d_logLuminance,
	unsigned int* const d_buckets,
	float min_logLum,
	float max_logLum,
	float range,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	// Get x and y coordinates
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	// Calculate the pixel id
	int id = idx + idy * numCols;

	// If the x or y coordinate is bigger than the number of columns or rows return as it is out of the image
	if (idx >= numCols || idy >= numRows) return;

	// Calculate the histogram bin and atomically increase it
	int histoIndex = (d_logLuminance[id] - min_logLum) / range * numBins;
	atomicAdd(&(d_buckets[histoIndex]), 1);
}

__global__ void Scan(unsigned int* const d_buckets,
	unsigned int* const d_cdf,
	const size_t numBins)
{
	extern __shared__ int cdfShared[];

	int id = threadIdx.x;

	// Load into shared memory
	cdfShared[id] = d_buckets[id];
	__syncthreads();

	int offset = 1;
	for (int counter = 0; counter < log2f(numBins); counter++)
	{
		// Get the value of the offset if the Id is less then the offset store 0
		unsigned int temp = 0;
		if (id >= offset)
		{
			temp = cdfShared[id - offset];
		}
		__syncthreads();

		// Add the value from the offset to the element
		cdfShared[id] += temp;
		__syncthreads();

		// Double the offset
		offset *= 2;
	}

	// Load all the cdf form shared memory
	if (id < numBins) d_cdf[id] = cdfShared[id];
}

//
// Device Memory
//
float* d_maxMinTemp;
float* d_max;
float* d_min;
unsigned int* d_buckets;

dim3 blockSize;
dim3 gridSize;
int gridSizeCol;
int gridSizeRows;
int blockSi;

void findMaxMin(const float* const d_logLuminance,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols)
{
	// Sizes for reducing the blocks
	int col2 = pow(2, ceil(log2((float)gridSizeCol)));
	int row2 = pow(2, ceil(log2((float)gridSizeRows)));
	const dim3 gridSizeReduce(col2, row2, 1);

	//
	// Find Max/Min
	// 	

	// Allocate temp storage for max/min reductions
	cudaMalloc((void**)&d_maxMinTemp, sizeof(float)* row2 * col2);

	// Find Max
	cudaMemset(d_maxMinTemp, -FLT_MAX, col2 * row2 * sizeof(float));
	findMaxMin << <gridSize, blockSize, sizeof(float)* blockSi * blockSi >> >(d_logLuminance,
		numRows,
		numCols,
		d_maxMinTemp,
		MAX,
		numRows * numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	float* temp = new float[row2 * col2];
	checkCudaErrors(cudaMemcpy(temp, d_maxMinTemp, sizeof(float)* row2 * col2, cudaMemcpyDeviceToHost));

	// Reduce Blocks
	findMaxMin << <1, gridSizeReduce, sizeof(float)* row2 * col2 >> >(d_maxMinTemp,
		row2,
		col2,
		d_maxMinTemp,
		MAX,
		gridSizeCol * gridSizeRows);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&max_logLum, d_maxMinTemp, sizeof(float), cudaMemcpyDeviceToHost));

	// Find Min
	cudaMemset(d_maxMinTemp, FLT_MAX, col2 * row2 * sizeof(float));
	findMaxMin << <gridSize, blockSize, sizeof(float)* blockSi * blockSi >> >(d_logLuminance,
		numRows,
		numCols,
		d_maxMinTemp,
		MIN,
		numRows * numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Reduce Blocks
	findMaxMin << <1, gridSizeReduce, sizeof(float)* row2 * col2 >> >(d_maxMinTemp,
		row2,
		col2,
		d_maxMinTemp,
		MIN,
		gridSizeCol * gridSizeRows);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(&min_logLum, d_maxMinTemp, sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_maxMinTemp);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	//TODO
	/*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	store in min_logLum and max_logLum
	2) subtract them to find the range
	3) generate a histogram of all the values in the logLuminance channel using
	the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	4) Perform an exclusive scan (prefix sum) on the histogram to get
	the cumulative distribution of luminance values (this should go in the
	incoming d_cdf pointer which already has been allocated for you)       */

	// Sizes for Blocks
	float numThreads = 32.0f;
	blockSi = (int)numThreads;
	gridSizeCol = std::ceil(numCols / numThreads);
	gridSizeRows = std::ceil(numRows / numThreads);
	blockSize = dim3(blockSi, blockSi, 1);
	gridSize = dim3(gridSizeCol, gridSizeRows, 1);


	//
	// Find Max Min
	//

	findMaxMin(d_logLuminance,
			   min_logLum,
			   max_logLum,
			   numRows,
			   numCols);

	//
	// Generate Histo
	//

	// Allocate memory for buckets
	cudaMalloc((void**)&d_buckets, sizeof(unsigned int)* numBins);

	// Default values to zero just in case
	cudaMemset(d_buckets, 0, numBins * sizeof(float));
	cudaMemset(d_cdf, 0, numBins * sizeof(float));

	generateHistogram << <gridSize, blockSize >> >(d_logLuminance,
												   d_buckets,
												   min_logLum,
												   max_logLum,
												   max_logLum - min_logLum,
												   numRows,
												   numCols,
												   numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	//
	// Generate CDF
	//
	Scan << <1, numBins, numBins * sizeof(float) >> >(d_buckets,
													  d_cdf,
													  numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	cudaFree(d_buckets);
}

