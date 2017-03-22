#include <chrono>
#include <cstdint>
#include <iostream>

using clock_type = std::chrono::high_resolution_clock;

int main()
{
    static const size_t nnz = 1UL << 32;

    double* a;
    posix_memalign((void**)(&a), 64, nnz * sizeof(double));

//#pragma omp parallel for
    for (size_t k = 0; k < nnz; k++) 
        a[k] = 1.25;

    for (int iter = 0; iter < 6; iter++) {
        auto start = clock_type::now();

        double res = 0.0;
#pragma omp parallel for reduction(+:res)
        for (size_t k = 0; k < nnz; k++) 
            res = res + 0.5 * a[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
        
        std::cout << (double)(2 * nnz) / (double)(ns.count()) << ", res = " << res << std::endl;
    }

    free(a);
}
