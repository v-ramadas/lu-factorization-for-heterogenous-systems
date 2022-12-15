#include "columnBlockFactorization.cuh"

//const int numStreams = 16;

__global__ void columnBlock_pivotKernel(unsigned int n, float* L, float* A, unsigned int index) {
    unsigned int pivotIdx = (threadIdx.x + blockIdx.x * blockDim.x);
    unsigned int iIdx = index * n;
    // Set diagonal of L to 1 and calculate pivot(which are the non diagonal entries of L)
    L[iIdx + index] = 1;
    if (pivotIdx > index && pivotIdx < n) {
        L[pivotIdx*n + index] = A[pivotIdx*n + index]/A[iIdx + index];
    }
}

__global__ void columnBlock_kernel (unsigned int n, float* L, float* U, unsigned int blockWidth, unsigned int iIndex, unsigned int jIndex)
{
    unsigned int pivotIdx = (threadIdx.x + blockIdx.x * blockDim.x)/blockWidth;
    unsigned int columnIdx = (threadIdx.x + blockIdx.x * blockDim.x) % blockWidth;
    unsigned int iIdx = iIndex * n;
    // Calculate U
    if (pivotIdx > iIndex && pivotIdx < n && (jIndex+columnIdx < n) && (jIndex+columnIdx >= iIndex) ) {
        U[pivotIdx * n + jIndex + columnIdx] -= L[pivotIdx*n + iIndex]*U[iIdx + jIndex + columnIdx];
    }
}

int columnBlock(unsigned int n, float* A, float* L, float* U, unsigned int blockWidth, unsigned int numStreams) {
    unsigned int threadsPerBlock = calculateThreadNum(n);
    unsigned int blocksPerGrid1 = (n + threadsPerBlock - 1)/threadsPerBlock;
    unsigned int blocksPerGrid2 = (n*blockWidth + threadsPerBlock - 1)/threadsPerBlock;

    cudaStream_t *stream = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    for (unsigned int i = 0; i < n; ++i) {
        columnBlock_pivotKernel <<< blocksPerGrid1, threadsPerBlock>>> (n, L, U, i);
        cudaError_t  status = cudaDeviceSynchronize();
        // Spawn multiple threads to schedule computation on multiple CUDA streams
#pragma omp parallel for schedule(static)
        for (unsigned int j = 0; j < n; j = j + blockWidth) {
            columnBlock_kernel <<< blocksPerGrid2, threadsPerBlock, 0, stream[(j/blockWidth)%numStreams] >>> (n, L, U, blockWidth, i, j);
            cudaDeviceSynchronize();
        }
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    delete[] stream;
    return 0;
}
