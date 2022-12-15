#include "luFactorization.hh"
#include "luFactorization.cuh"
#include "utils.cuh"

void printMatrix(int h, int w, float *A)
{
    std::printf("[\n");
    for (int i = 0; i < h; ++i) {
        std::printf("[");
        for (int j = 0; j < w; ++j) {
            std::printf("%f ", A[i*w + j]); 
        }
        std::printf("]\n");
    }
    std::printf("]\n");
}

int calculateThreadNum ( int n) {
    if (n < 1024) return n;
    else return 1024;
}

void readArray(const char* fileName, int h, int w, float* A)
{
    std::ifstream f(fileName);
    if (f.is_open()) {
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                f >> A[r*h + c];
            }
        }
    } else {
        std::printf("Failed to load from %s\n", fileName);
    }
}

int compareArray(int h, int w, float* A, float* B)
{
    const float threshold = 1e-3;
    int result = 0;
    for (int i = 0; i < h*w; ++i) {
        if (A[i] - B[i] > threshold) {
            result = 1;
            std::printf("At rdx : %d, cdx : %d, with vals %f, %f\n", i/w, i%w, A[i], B[i]);
            break;
        }
    }
    return result;
}
