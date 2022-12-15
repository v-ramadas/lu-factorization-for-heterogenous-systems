#pragma once
#include <fstream>

void printMatrix(int h, int w, float *A);

int calculateThreadNum ( int n);

void readArray(const char* fileName, int h, int w, float* A);

//int compareArray(const char* fileName, int h, int w, float* A);

int compareArray(int h, int w, float*A, float* B);
