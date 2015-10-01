#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

// Matrix Fields
#define RANDRANGE  2000   
#define EQUAL_TOL  .0001f   
#define MAX_SIDE_LENGTH 10
#define MAX_TILE_SIDE_LENGTH 6
#define THREADS_PER_BLOCK 256

__global__ void matrixMultKernel(float* Md, float* Nd, float* Pd, int Width, int TileWidth) 
{ 
	extern __shared__ float shared[];
	float* Mds = (float*)shared;
	float* Nds = (float*)&Mds[TileWidth * TileWidth];

	int bx = blockIdx.x;  
	int by = blockIdx.y; 
	int tx = threadIdx.x; 
	int ty = threadIdx.y;  
	
	// Identify the row and column of the Pd element to work on 
	int Row = by * TileWidth + ty;
	int Col = bx * TileWidth + tx;
	float Pvalue = 0; 
	
	// Loop over the Md and Nd tiles required to compute the Pd element 
	for (int m = 0; m < Width / TileWidth; ++m)
	{ 
		// Collaborative loading of Md and Nd tiles into shared memory    
		Mds[ty + TileWidth * tx] = Md[Row*Width + (m*TileWidth + tx)];
		Nds[ty + TileWidth * tx] = Nd[Col + (m*TileWidth + ty)*Width];
		
		__syncthreads();  
		
		for (int k = 0; k < TileWidth; ++k) Pvalue += Mds[ty + TileWidth * k] * Nds[k + TileWidth * tx];
		
		__syncthreads(); 
	} 
	
	Pd[Row*Width+Col] = Pvalue; 
}   

__global__ void dotProductKernel(float* Md, float* Nd, float* Pd)
{
	int localIdx = threadIdx.x;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int offset = blockDim.x / 2;

	// Load Multapliation into shared memory
	extern __shared__ float temp[];
	temp[localIdx] = Md[idx] * Nd[idx];

	__syncthreads();

	// Sum products
	while (offset != 0)
	{		
		if (localIdx < offset) temp[localIdx] += temp[localIdx + offset];

		offset /= 2;
		
		__syncthreads();
	}

	// Set the result of the sums
	if (localIdx == 0) Pd[blockIdx.x] = temp[0];
}

__global__ void sumReduceKernel(float* Md, float* Pd)
{
	int localIdx = threadIdx.x;
	int idx = localIdx + blockDim.x * blockIdx.x;
	int offset = blockDim.x / 2;

	// Load Blocks portion of memory into shared memory
	extern __shared__ float temp[];
	temp[threadIdx.x] = Md[idx];

	__syncthreads();

	// Sum up everything for the block
	while (offset != 0)
	{
		if (localIdx < offset) temp[localIdx] += temp[localIdx + offset];

		__syncthreads();		

		offset /= 2;
	}

	// Set result
	if (localIdx == 0) Pd[blockIdx.x] = temp[0];
}


// Allocates a matrix with random float entries. 
void randomInit(float* data, int size) 
{    
	for (int i = 0; i < size; ++i) data[i] = (float)((rand() % RANDRANGE + 1) / (float)RANDRANGE);
}   
	
//
// Determine if two floats are within a tolerance of each other
//
bool EqaulWithTol(float a, float b, float tol = EQUAL_TOL)
{
	return abs(a - b) <= tol;
}

float MatrixMult(float* h_matrixA, float* h_matrixB, float* h_outMatrix, int matrixSideLength, int tileWidth)
{
	unsigned int mem_size = sizeof(float)* matrixSideLength * matrixSideLength;

	//  allocate device memory    
	float* d_A;
	float* d_B;
	float* d_Out;
	cudaMalloc((void**)&d_A, mem_size);
	cudaMalloc((void**)&d_B, mem_size);
	cudaMalloc((void**)&d_Out, mem_size);

	//  copy host memory to device    
	cudaMemcpy(d_A, h_matrixA, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_matrixB, mem_size, cudaMemcpyHostToDevice);

	dim3 blocks(tileWidth, tileWidth);
	dim3 grid(matrixSideLength / tileWidth, matrixSideLength / tileWidth);
	
	// Start the timer
	auto start = std::chrono::high_resolution_clock::now();
	
	// execute the kernel    
	matrixMultKernel <<< grid, blocks, sizeof(float)* tileWidth  * tileWidth * 2 >>>(d_A, d_B, d_Out, matrixSideLength, tileWidth);
	cudaDeviceSynchronize();

	// Stop the timer
	auto finish = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> elapsed_seconds = finish - start;
	
	// Calculate the number of flops
	float FLops = (float)((7 + 12 * (matrixSideLength / tileWidth) + 5 * matrixSideLength)  * (matrixSideLength * matrixSideLength)) / (float)((elapsed_seconds.count()));

	// print results
	std::cout << std::setw(16) << elapsed_seconds.count() << std::setw(12) << FLops / 1000000000.0f;

	// copy result from device to host    
	cudaMemcpy(h_outMatrix, d_Out, mem_size, cudaMemcpyDeviceToHost);

	// Free cuda memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_Out);

	return FLops;
}

//
// Calculate the Dot Product of d_vectorA * d_vectorB stores result in d_outVector
//
float DotProduct(float* h_vectorA, float* h_vectorB, int vectorLength)
{
	// allocate host memory for matrices A and B    
	unsigned int mem_size = sizeof(float)* vectorLength;	

	// Determine number of Blocks
	int blocks = THREADS_PER_BLOCK;
	int grid = vectorLength / blocks;
	if (vectorLength < blocks)
	{
		blocks = vectorLength;
		grid = 1;
	}

	float* h_outVector = (float*)malloc(sizeof(float)* grid);
	memset(h_outVector, 0, sizeof(float) * grid);

	// allocate device memory    
	float* d_A;
	float* d_B;
	float* d_Out;

	// Allocate cuda memory
	cudaMalloc((void**)&d_A, mem_size);
	cudaMalloc((void**)&d_B, mem_size);
	cudaMalloc((void**)&d_Out, grid * sizeof(float));

	// copy host memory to device    
	cudaMemcpy(d_A, h_vectorA, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_vectorB, mem_size, cudaMemcpyHostToDevice);


	// execute the kernel    
	dotProductKernel << <grid, blocks, sizeof(float) * blocks >> >(d_A, d_B, d_Out);
	cudaDeviceSynchronize();

	// Reduce if needed
	if (grid > 1)
	{
		sumReduceKernel << <1, grid, sizeof(float) * grid >> >(d_Out, d_Out);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(h_outVector, d_Out, grid * sizeof(float), cudaMemcpyDeviceToHost);

	// Free cuda memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_Out);


	float returnValue = h_outVector[0];
	free(h_outVector);

	return returnValue;
}

bool testMatrix(float* MatrixA, float* MatrixB, float* Matrix, int matrixSideLength)
{
	// Generate test index
	int idX = rand() % matrixSideLength;
	int idY = rand() % matrixSideLength;

	// Get test value from matrix
	float testValue = Matrix[idX * matrixSideLength + idY];

	// Allocate memory for vectors
	float* h_vectorA = (float*)malloc(sizeof(float)* matrixSideLength);
	float* h_vectorB = (float*)malloc(sizeof(float)* matrixSideLength);	

	// Load row andf column into vectors
	for (int i = 0; i < matrixSideLength; i++)
	{
		h_vectorA[i] = MatrixA[idX * matrixSideLength + i];
		h_vectorB[i] = MatrixB[idY + matrixSideLength * i];
	}

	// Find the dot product
	float dotProduct = DotProduct(h_vectorA, h_vectorB, matrixSideLength);

	bool retValue = true;

	std::cout << std::setw(11) << std::to_string(idX) + "," + std::to_string(idY) << std::setw(15) << testValue << std::setw(15) << dotProduct;
	if (EqaulWithTol(dotProduct, testValue))
	{
		std::cout << std::setw(10) << "Passed";
	}
	else
	{
		std::cout << std::setw(10) << "Failed";
		retValue = false;
	}

	free(h_vectorA);
	free(h_vectorB);

	return retValue;
}

/////////////////////////////////////////////////////////// Program main /////////////////////////////////////////////////////////    
int main(int argc, char** argv) 
{	
	// set seed for rand()   
	srand(2456);

	float* h_A;
	float* h_B;
	float* h_C;

	bool testFailed = false;
	float maxFlops = 0;
	int maxSide = 0;
	int maxTile = 0;

	std::cout << "Comparison Tolerance: " << EQUAL_TOL << std::endl;
	std::cout << std::setw(5) << "N" << std::setw(12) << "Tile Width" << std::setw(16) << "Time" << std::setw(12) << "FLOPs" << std::setw(11) << "Test Index" << std::setw(15) << "Matrix Value" << std::setw(15) << "Vector Value" << std::setw(10) << "Test" << "\n";

	for (int i = 1; i < MAX_SIDE_LENGTH; i++)
	{
		int matrixSideLength = 1<<i;

		// allocate host memory for matrices A and B    
		unsigned int mem_size = sizeof(float)* matrixSideLength * matrixSideLength;
		h_A = (float*)malloc(mem_size);
		h_B = (float*)malloc(mem_size);
		h_C = (float*)malloc(mem_size);

		// initialize host memory    
		randomInit(h_A, matrixSideLength * matrixSideLength);
		randomInit(h_B, matrixSideLength * matrixSideLength);		

		for (int x = 0; x < MAX_TILE_SIDE_LENGTH; x++)	
		{
			// only run multiply where the tile size is smaller than the matrix size
			if (x > i) break;

			int tileSize = 1 << x;

			// Print out configuration of this run
			std::cout << std::setw(5) << matrixSideLength << std::setw(12) << tileSize;

			memset(h_C, 0, mem_size);
			float flops = MatrixMult(h_A, h_B, h_C, matrixSideLength, tileSize);

			// Check if this run was faster than the fastest
			if (flops > maxFlops && !std::isinf(flops))
			{
				maxFlops = flops;
				maxSide = matrixSideLength;
				maxTile = tileSize;
			}

			// Test the result of the multiply
			if (!testMatrix(h_A, h_B, h_C, matrixSideLength)) testFailed = true;

			// start new row for table
			std::cout << "\n";
			
		}

		// Free host memory
		free(h_A);
		free(h_B);
		free(h_C);
	}

	if (!testFailed) printf("\nAll Tests Passed\n");
	else printf("Some Tests Failed\n");
	
	printf("Best GFlops %5.5f\nBest Side Size %d\nBest Tile Size: %d\n", maxFlops / 1000000000.0f, maxSide, maxTile);

	system("pause");
}       