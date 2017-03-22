#include <chrono>
#include <cstdlib>
#include <iostream>

using clock_type = std::chrono::high_resolution_clock;

int main()
{
    static const size_t n = 1UL << 30; // 1G elements

    double* restrict a = nullptr;
    posix_memalign((void**)(&a), 64, n * sizeof(double));

#pragma omp parallel for
    for (size_t k = 0; k < n; k++) 
        a[k] = 0.0;

    std::cout << "1D-FMADD experiment:" << std::endl;
    for (int exp = 0; exp < 6; exp++) {
        auto start = clock_type::now();

        double res = 0.0;
#pragma omp parallel for reduction(+:res)
        for (size_t k = 0; k < n; k++)
            res = res + 0.5 * a[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

        std::cout
            << "read bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "total bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "performance: " << (double)(2 * n) / (double)(ns.count()) << " [GFLOP/s]"
            << ", result = " << res
            << std::endl;
    }
    std::cout << std::endl;

    free(a);
}
