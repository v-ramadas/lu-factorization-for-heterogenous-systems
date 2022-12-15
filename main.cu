#include "luFactorization.cuh"
#include "columnBlockFactorization.cuh"
#include <chrono>
#include <stdlib.h>
#include <random>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

using namespace std;

int main(int argc, char** argv)
{
    high_resolution_clock::time_point start_serial;
    high_resolution_clock::time_point end_serial;
    duration<double, std::milli> duration_sec;

    int choice = std::atoi(argv[1]);
    unsigned int blockSize = std::atoi(argv[2]);
    int n;
    float* A;
    float B[] = {8, 4, -2, 3, 4, 9, 2, -5, -2, 2, 5, -2, 3, -5, -2, 7};

    if (choice == 0) {
        n = 4;
        A = new float[n*n];
        memcpy(A, B, n*n*sizeof(float));
    } else {
        n = choice;
        A = new float [n*n];
        int seed = 1;
        std::mt19937 generator(seed);
        const float minval = 0, maxval = 10;
        std::uniform_real_distribution<float> dist(minval, maxval);
        for (int i = 0; i < n*n; ++i) {
            A[i] = dist(generator);
        }
    }
    float *L, *U, *goldenL, *goldenU;
    L = new float[n*n];
    U = new float[n*n];
    goldenL = new float[n*n];
    goldenU = new float[n*n];

/*
 * Serial Implementation
 */
    start_serial = high_resolution_clock::now();
    int status = luFactorizationSerial(n, A, L, U);
    end_serial = high_resolution_clock::now();
    std::copy(L, L + n*n, goldenL);
    std::copy(U, U + n*n, goldenU);
    if (1 == status) {
        printf("Non-singular matrix\n");
        return status;
    }
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>> (end_serial - start_serial);
    if (choice == 0) {
        printf("A:\n");
        printMatrix(n, n, A);
        printf("Serial L:\n");
        printMatrix(n, n, L);
        printf("U:\n");
        printMatrix(n, n, U);
    }
    printf("Serial Time: %lf\n", duration_sec.count());

/*
 * OMP Implementation
 */
    memset(L, 0, n*n*sizeof(float));
    memset(U, 0, n*n*sizeof(float));
    double start_omp = omp_get_wtime();
    status = luFactorizationOMP(n,A, L, U);
    double end_omp = omp_get_wtime();

    printf("OMP Time: %lf\n", (end_omp - start_omp)*1000);
    if (choice == 0) {
        printf("OMP L:\n");
        printMatrix(n, n, L);
        printf("U:\n");
        printMatrix(n, n, U);
    } else {
        int res1 = compareArray(n, n, L, goldenL);
        int res2 = compareArray(n, n, U, goldenU);
        if (res1 || res2) {
            printf ("Failed!\n");
        } else {
            printf("Passed!\n");
        }
    }

/*
 * CUDA Implementation
 */
    float *dU, *dL;
    float ms;
    cudaEvent_t start_cuda;
    cudaEvent_t stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    cudaMalloc((float**)&dU, n*n*sizeof(float));
    cudaMalloc((float**)&dL, n*n*sizeof(float));
    cudaMemcpy(dU, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dL, 0, n*n*sizeof(float));

    cudaEventRecord(start_cuda);
    luFactorizationCUDA(n, A, dL, dU);
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);
    cudaEventElapsedTime(&ms, start_cuda, stop_cuda);

    printf("CUDA Time: %lf\n", ms);
    cudaMemcpy(U, dU, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, dL, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dL);
    cudaFree(dU);
    if (choice == 0) {
        printf("CUDA L:\n");
        printMatrix(n, n, L);
        printf("U:\n");
        printMatrix(n, n, U);
    } else {
        int res1 = compareArray(n, n, L, goldenL);
        int res2 = compareArray(n, n, U, goldenU);
        if (res1 || res2) {
            printf ("Failed!\n");
        } else {
            printf("Passed!\n");
        }
    }


    delete[] L;
    delete[] U;
    delete[] A;

/*
 * Hybrid Implementation
 */
    cudaMallocManaged(&A, n*n*sizeof(float));
    cudaMallocManaged(&U, n*n*sizeof(float));
    cudaMallocManaged(&L, n*n*sizeof(float));

    if (choice == 0) {
        n = 4;
        A = new float[n*n];
        memcpy(A, B, n*n*sizeof(float));
    } else {
        n = choice;
        A = new float [n*n];
        int seed = 1;
        std::mt19937 generator(seed);
        const float minval = 0, maxval = 10;
        std::uniform_real_distribution<float> dist(minval, maxval);
        for (int i = 0; i < n*n; ++i) {
            A[i] = dist(generator);
        }
    }

    std::copy(A, A + n*n, U);
    cudaEventRecord(start_cuda);
    luFactorizationHybrid(n, A, L, U);
    cudaEventSynchronize(stop_cuda);
    cudaEventRecord(stop_cuda);
    cudaEventElapsedTime(&ms, start_cuda, stop_cuda);

    printf("Hybrid Time : %lf\n", ms);
    if (choice == 0) {
        printf("Hybrid L:\n");
        printMatrix(n, n, L);
        printf("U:\n");
        printMatrix(n, n, U);
    } else {
        int res1 = compareArray(n, n, L, goldenL);
        int res2 = compareArray(n, n, U, goldenU);
        if (res1 || res2) {
            printf ("Failed!\n");
        } else {
            printf("Passed!\n");
        }
    }

/*
 * Column Block Implementation
 */
    std::copy(A, A + n*n, U);
    memset(L, 0, n*n*sizeof(float));
    cudaEventRecord(start_cuda);
    columnBlock(n, A, L, U, blockSize);
    cudaEventSynchronize(stop_cuda);
    cudaEventRecord(stop_cuda);
    cudaEventElapsedTime(&ms, start_cuda, stop_cuda);

    printf("Column Block Time : %lf\n", ms);
    if (choice == 0) {
        printf("Column Block L:\n");
        printMatrix(n, n, L);
        printf("U:\n");
        printMatrix(n, n, U);
    } else {
        int res1 = compareArray(n, n, L, goldenL);
        int res2 = compareArray(n, n, U, goldenU);
        if (res1 || res2) {
            printf ("Failed! %d, %d\n", res1, res2);
        } else {
            printf("Passed!\n");
        }
    }

    cudaFree(A);
    cudaFree(U);
    cudaFree(L);
    delete[] goldenL;
    delete[] goldenU;
    return status;
}
