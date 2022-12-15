#pragma once
#include "luFactorization.hh"
__host__ int luFactorizationCUDA(long int n, float *A, float *L, float *U);
__global__ void luFactorizationCUDA_pivotKernel(long int n, float *U, float *L, long int index);
__global__ void luFactorizationCUDA_kernel(long int n, float *A, float *L, float *U, long int index);



__host__ int luFactorizationHybrid(long int n, float *A, float *L, float *U, unsigned int numStreams);
__global__ void luFactorizationHybrid_kernel(long int n, float *A, float *L, float *U, long int index);
