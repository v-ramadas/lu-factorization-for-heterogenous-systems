#include "luFactorization.cuh"
#include <stdlib.h>
#include <stdio.h>

__global__ void luFactorizationCUDA_pivotKernel(long int n, float *U, float* L, long int index)
{
    long int pivotIdx = threadIdx.x + blockIdx.x * blockDim.x;
    long int iIdx = index * n;
    // Setting diagonal of L to 1, calculating pivot (which is also L's non diagonal entries)
    L[iIdx + index] = 1; 
    if (pivotIdx > index && pivotIdx < n) {
        L[pivotIdx*n + index] = U[pivotIdx*n + index]/U[iIdx + index];
    }
}

__global__ void luFactorizationCUDA_kernel(long int n, float *A, float *L, float *U, long int index)
{
    long int arrayElement = threadIdx.x + blockIdx.x * blockDim.x;
    long int pivotIdx = arrayElement / n;
    long int columnIdx = arrayElement % n;
   	long int iIdx = index * n;

    // Calculating U
    if ((pivotIdx > index) && (columnIdx >= index) && (pivotIdx < n)) {
        U[arrayElement] -= L[pivotIdx*n + index] * U[iIdx + columnIdx];
//        L[arrayElement] += pivot[pivotIdx] * L[iIdx + columnIdx];
    }
}


int luFactorizationCUDA(long int n, float *A, float *L, float *U) {
    long int threadsPerBlock = calculateThreadNum(n);
    long int blocksPerGrid1 = (n + threadsPerBlock - 1)/threadsPerBlock; 
    long int blocksPerGrid2 = (n*n + threadsPerBlock - 1)/threadsPerBlock; 
    for (long int i = 0; i < n; ++i) {
        luFactorizationCUDA_pivotKernel <<< blocksPerGrid1, threadsPerBlock>>> (n, U, L, i);
        luFactorizationCUDA_kernel <<< blocksPerGrid2, threadsPerBlock>>> (n, A, L, U, i);
        cudaDeviceSynchronize();
    }
    return 0;
}
