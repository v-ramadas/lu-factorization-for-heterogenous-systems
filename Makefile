CFLAGS = -Wall -O3  -std=c++17 -fopenmp -fopt-info-vec -march=native
CUDAFLAGS = -Xcompiler -Wall -Xcompiler -O3  -Xptxas -O3 -std=c++17 -Xcompiler -fopenmp -Xcompiler -fopt-info-vec -Xcompiler -march=native
LFLAGS = -lstdc++
CC = gcc
NVCC = nvcc
luFactorization: luFactorizationSerial.cc luFactorizationOMP.cc luFactorizationCUDA.cu luFactorizationHybrid.cu columnBlockFactorization.cu utils.cu main.cu
	$(NVCC) $(CUDAFLAGS) $? $(LFLAGS) -o $@

clean:
	rm -f *.o luFactorization luFactorizationCUDA
