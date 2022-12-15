#include <cstdio>
#include <fstream>
using namespace std;

void readArray(const char* fileName, int h, int w, float* A)
{
    ifstream f(fileName);
    if (f.is_open())
    {
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                f >> A[r*h + c];
            }
        }
    }
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            std::printf("%f, ", A[r*h+c]);
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    int n = std::atoi(argv[1]);
    float* A = new float[n];
    readArray("LMatrix_n6.bin", n, n, A);
    delete[] A;
}
