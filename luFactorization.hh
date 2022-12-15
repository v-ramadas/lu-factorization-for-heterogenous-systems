#pragma once
#include <cstddef>
#include <cstdio>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.cuh"

int luFactorizationSerial(long int n, float *A, float *L, float *U);

int luFactorizationOMP(long int n, float *A, float *L, float *U);
