#include "luFactorization.cuh"
#include <stdlib.h>
#include <stdio.h>

//const int numStreams = 16;

__global__ void luFactorizationHybrid_kernel(long int n, float* A, float* L, float* U, float pivot, long int iIndex, long int jIndex)
{
    long int iIdx = iIndex * n;
    long int jIdx = jIndex * n;
    long int columnIdx = threadIdx.x + (blockIdx.x*blockDim.x);
    // Setting diagonal of L to 1
    L[iIdx + iIndex] = 1;
    __syncthreads();

    // Calculating L, U
    if ((columnIdx >= iIndex) && (columnIdx < n)) {
        U[jIdx + columnIdx] = U[jIdx + columnIdx] - pivot * U[iIdx + columnIdx];
        L[jIdx + columnIdx] = L[jIdx + columnIdx] + pivot * L[iIdx + columnIdx];
    }
    __syncthreads();
}

__host__ int luFactorizationHybrid(long int n, float* A, float* L, float* U,unsigned int numStreams)
{
    int threadsPerBlock = calculateThreadNum(n);
    int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;
    std::printf("Here\n");
    cudaStream_t* stream = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    for (long int i = 0; i < n; ++i) {
       long int iIdx = i * n;
       // Spawn multiple threads to schedule CUDA instructions on different streams
#pragma omp parallel for schedule(static) 
       for (long int j = i+1; j < n; ++j) {
           long int jIdx = j * n;
           float pivot = U[jIdx + i]/U[iIdx + i];
           luFactorizationHybrid_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream[j%numStreams]>>> (n, A, L, U, pivot, i, j);
       }

       cudaDeviceSynchronize();
    }
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    delete[] stream;
    return 0;
}
