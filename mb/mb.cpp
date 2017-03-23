#include <chrono>
#include <cstdlib>
#include <iostream>

#include <abhsf/utils/restrict.h>

using clock_type = std::chrono::high_resolution_clock;

int main()
{
 // static const size_t n = 16UL * (1UL << 20); // 16M elements
 // static const size_t n = 1UL << 30; // 1G elements
    static const size_t n = 1UL << 28; // 256M elements
    static const int n_exp = 4;

    double* RESTRICT a = nullptr;
    double* RESTRICT b = nullptr;
    double* RESTRICT c = nullptr;

    posix_memalign((void**)(&a), 64, n * sizeof(double));
    posix_memalign((void**)(&b), 64, n * sizeof(double));
    posix_memalign((void**)(&c), 64, n * sizeof(double));

#pragma omp parallel for
    for (size_t k = 0; k < n; k++) {
        a[k] = 0.0;
        b[k] = 0.5;
        c[k] = 0.5;
    }

    std::cout << "COPY experiment:" << std::endl;
    for (int exp = 0; exp < n_exp; exp++) {
        auto start = clock_type::now();

#pragma omp parallel for
        for (size_t k = 0; k < n; k++)
            a[k] = b[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

        std::cout
            << "read bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "write bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "total bw: " << (double)(2 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s]"
            << std::endl;
    }
    std::cout << std::endl;

    std::cout << "ADD experiment:" << std::endl;
    for (int exp = 0; exp < n_exp; exp++) {
        auto start = clock_type::now();

#pragma omp parallel for
        for (size_t k = 0; k < n; k++)
            a[k] = b[k] + c[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

        std::cout
            << "read bw: " << (double)(2 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "write bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "total bw: " << (double)(3 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "performance: " << (double)n / (double)(ns.count()) << " [GFLOP/s]"
            << std::endl;
    }
    std::cout << std::endl;

    std::cout << "FMADD experiment:" << std::endl;
    for (int exp = 0; exp < n_exp; exp++) {
        auto start = clock_type::now();

#pragma omp parallel for
        for (size_t k = 0; k < n; k++)
            a[k] = a[k] + b[k] * c[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

        std::cout
            << "read bw: " << (double)(3 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "write bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "total bw: " << (double)(4 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "performance: " << (double)(2 * n) / (double)(ns.count()) << " [GFLOP/s]"
            << std::endl;
    }
    std::cout << std::endl;

    std::cout << "1D-ADD experiment:" << std::endl;
    for (int exp = 0; exp < n_exp; exp++) {
        auto start = clock_type::now();

        double res = 0.0;
#pragma omp parallel for reduction(+:res)
        for (size_t k = 0; k < n; k++)
            res = res + a[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

        std::cout
            << "read bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "total bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "performance: " << (double)n / (double)(ns.count()) << " [GFLOP/s]"
            << ", result = " << res 
            << std::endl;
    }
    std::cout << std::endl;

    std::cout << "1D-FMADD experiment:" << std::endl;
    for (int exp = 0; exp < n_exp; exp++) {
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

    std::cout << "2D-FMADD experiment:" << std::endl;
    for (int exp = 0; exp < n_exp; exp++) {
        auto start = clock_type::now();

        double res = 0.0;
#pragma omp parallel for reduction(+:res)
        for (size_t k = 0; k < n; k++)
            res = res + a[k] * b[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

        std::cout
            << "read bw: " << (double)(2 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "total bw: " << (double)(2 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "performance: " << (double)(2 * n) / (double)(ns.count()) << " [GFLOP/s]"
            << ", result = " << res
            << std::endl;
    }
    std::cout << std::endl;

    std::cout << "3D-FMADD experiment:" << std::endl;
    for (int exp = 0; exp < n_exp; exp++) {
        auto start = clock_type::now();

#pragma omp parallel for
        for (size_t k = 0; k < n; k++)
            c[k] = c[k] + a[k] * b[k];

        auto diff = clock_type::now() - start;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

        std::cout
            << "read bw: " << (double)(3 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "write bw: " << (double)(n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "total bw: " << (double)(4 * n * sizeof(double)) / (double)(ns.count()) << " [GB/s], "
            << "performance: " << (double)(2 * n) / (double)(ns.count()) << " [GFLOP/s]"
            << std::endl;
    }
    std::cout << std::endl;

    free(a);
    free(b);
    free(c);
}
