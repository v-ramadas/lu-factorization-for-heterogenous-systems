#pragma once
#include <cstddef>
#include <cstdio>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.cuh"

__global__ void columnBlock_pivotKernel(unsigned int n, float* L, float* U, unsigned int index);
__global__ void columnBlock_kernel (unsigned int n, float* L, float* U, unsigned int blockWidth, unsigned int iIndex, unsigned int jIndex);
int columnBlock(unsigned int n, float* A, float* L, float* U, unsigned int blockWidth, unsigned int numStreams);
