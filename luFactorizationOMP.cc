#include "luFactorization.hh"

int luFactorizationOMP(long int n, float *A, float *L, float *U)
{
    long int pivotIdx = 0;
    float pivot;
    std::copy(A, A + n*n, U);
    // Setting diagonal of L to 1
#pragma omp parallel for schedule(static, 64)
    for (long int i = 0; i < n; ++i) {
        L[i*n + i] = 1;
    }

    // Calculating pivot and U
    for (long int i = 0; i < n; ++i) {
       long int iIdx = i * n;
#pragma omp parallel for schedule(static)
       for (long int j = i+1; j < n; ++j) {
           long int jIdx = j * n;
           pivot = U[jIdx + pivotIdx]/U[iIdx + pivotIdx];
           L[jIdx + i] = pivot;
           for (long int k = i; k < n; ++k) {
               U[jIdx + k] = U[jIdx + k] - pivot*U[iIdx + k];
//               L[jIdx + k] = L[jIdx + k] + pivot*L[iIdx + k];
           }
       }
       pivotIdx++;
    }
    return 0;
}
