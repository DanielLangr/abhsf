#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <mkl.h>

#include <abhsf/utils/colors.h>
#include <abhsf/utils/matrix_properties.h>
#include <abhsf/utils/matrix_market_reader.h>
#include <abhsf/utils/timer.h>

#ifdef __INTEL_COMPILER
    #define RESTRICT restrict
#else
    #define RESTRICT __restrict__
#endif

#define PRINT_RESULT(label) \
    do { \
        std::cout << std::left << std::setw(36) << label \
            << yellow << std::right << std::fixed << std::setprecision(1) << std::setw(10) \
            << n_mflops / timer.seconds() << reset << ", result = " << green << std::setprecision(3) \
            << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl; \
    } while (0)

static const int warmup_iters = 4;
static const int n_iters = 20;

using timer_type = chrono_timer<>;

using real_type = double;
using index_type = MKL_INT;

static_assert(sizeof(MKL_INT) == 4, "Size of MKL_INT is not 32 bits");

using element_t = std::tuple<index_type, index_type, real_type, uint64_t>;
using elements_t = std::vector<element_t>;

uint64_t morton(uint32_t a, uint32_t b)
{
    uint64_t c = 0;

    for (int i = 0; i < 32; i++)
        c |= (((uint64_t)a & (1UL << i)) << i) | (((uint64_t)b & (1UL << i)) << (i + 1));

    return c;
}

class coo_matrix
{
    public:
        void from_elements(elements_t& elements, const matrix_properties& props)
        {
/*
            // Morton ordering
            for (auto& element : elements)
                std::get<3>(element) = morton(std::get<0>(element), std::get<1>(element));
            std::sort(elements.begin(), elements.end(), 
                    [](const element_t& lhs, const element_t& rhs) {
                        return std::get<3>(lhs) < std::get<3>(rhs);
                    }
            );
*/
            // lexicographical ordering
            std::sort(elements.begin(), elements.end());

            m_ = props.m;
            n_ = props.n;
            const size_t nnz = elements.size();

            a_.resize(nnz);
            ia_.resize(nnz);
            ja_.resize(nnz);

            for (size_t k = 0; k < nnz; k++) {
                a_[k] = std::get<2>(elements[k]);
                ia_[k] = std::get<0>(elements[k]);
                ja_[k] = std::get<1>(elements[k]);
            }
        }

        void spmv_data_race(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            const size_t nnz = a_.size();

#pragma omp parallel for
            for (size_t k = 0; k < nnz; k++)
                y[ia_[k]] += a_[k] * x[ja_[k]];
        }

        void spmv_atomic_updates(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            const size_t nnz = a_.size();

#pragma omp parallel for
            for (size_t k = 0; k < nnz; k++)
#pragma omp atomic update
                y[ia_[k]] += a_[k] * x[ja_[k]];
        }

        void spmv_mkl(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            static const char transa = 'N';
            static const double alpha = 1.0;
            static const char matdescra[6] = { 'G', ' ', ' ', 'C', ' ', ' ' };

            const MKL_INT nnz = a_.size();

            mkl_dcoomv(&transa, &m_, &n_, &alpha, matdescra, 
                    a_.data(), ia_.data(), ja_.data(), &nnz, x, &alpha, y);
        }

        void release()
        {
            a_.clear();
            a_.shrink_to_fit();
            ia_.clear();
            ia_.shrink_to_fit();
            ja_.clear();
            ja_.shrink_to_fit();
        }

    private:
        std::vector<real_type> a_;
        std::vector<MKL_INT> ia_;
        std::vector<MKL_INT> ja_;
        MKL_INT m_, n_;
};

template <typename Iter>
double result(Iter begin, Iter end)
{
    double res = 0.0;
    while (begin != end) {
        res += *begin * *begin;
        ++begin;
    }
    return sqrt(res);
}

int main(int argc, char* argv[])
{
    int mattype = atoi(argv[1]);

    elements_t elements;
    matrix_properties props;

    if (mattype == 0) {
        // dense matrix
        std::cout << "Matrix: " << yellow << "DENSE" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 30000;
        props.nnz = props.m * props.n;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            for (size_t j = 0; j < props.n; j++)
                elements.emplace_back(i, j, 1.0, 0);
    }
    else if (mattype == 1) {
        // diagonal matrix
        std::cout << "Matrix: " << yellow << "DIAGONAL" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 500000000L;
        props.nnz = props.m;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            elements.emplace_back(i, i, 1.0, 0);
    }
    else if (mattype == 2) {
        // random-column "permutation" matrix
        std::cout << "Matrix: " << yellow << "RANDOM-COLUMN" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 500000000L;
        props.nnz = props.m;

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, props.n - 1);

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            elements.emplace_back(i, dist(mt), 1.0, 0);
    }
    else if (mattype == 3) {
        // L matrix
        std::cout << "Matrix: " << yellow << "L" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
     // props.m = props.n = 400000000L;
        props.m = props.n =  50000000L;
        props.nnz = props.m + props.n - 1;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m - 1; i++)
            elements.emplace_back(i, 0, 1.0, 0);
        for (size_t j = 0; j < props.n; j++)
            elements.emplace_back(props.m - 1, j, 1.0, 0);
    }
    else if (mattype == 4) {
        // single-column matrix
        std::cout << "Matrix: " << yellow << "SINGLE-COLUMN" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 500000000L;
        props.nnz = props.m;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            elements.emplace_back(i, 0, 1.0, 0);
    }
    else if (mattype == 5) {
        // single-row matrix
        std::cout << "Matrix: " << yellow << "SINGLE-ROW" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
     // props.m = props.n = 500000000L;
        props.m = props.n =  50000000L;
        props.nnz = props.m;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.n; i++)
            elements.emplace_back(props.m - 1, i, 1.0, 0);
    }
    else 
        throw std::runtime_error("Matrix type has not been specified!");

    std::cout << "COO matrix memory footprint: " << green
        << (double)(props.nnz * (8 + 8)) / (1024.0 * 1024.0 * 1024.0) 
        << reset << " [GB]" << std::endl;

    std::cout << "Nonzeros: " << magenta << std::right << std::setw(20) << props.nnz << reset << std::endl; 
    std::cout << "sizeof(MKL_INT): " << cyan << sizeof(MKL_INT) << reset << std::endl;

    real_type *x, *y;
    posix_memalign((void**)(&x), 64, props.n * sizeof(real_type));
    posix_memalign((void**)(&y), 64, props.m * sizeof(real_type));
    std::fill(x, x + props.n, 0.5);
    std::fill(y, y + props.m, 0.0);

    // verification
    for (const auto& element : elements) 
        y[std::get<0>(element)] += std::get<2>(element) * x[std::get<1>(element)];
    std::cout << "Expected result = " << green << result(y, y + props.n) << reset << std::endl;

    const double n_mflops = (double)(props.nnz * 2 * n_iters) / 1.0e6;

    // COO
    coo_matrix coo;
    coo.from_elements(elements, props);

    // naive COO - w/ data race
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        coo.spmv_data_race(x, y);
    timer_type timer(timer_type::start_now);
    for (int iter = 0; iter < n_iters; iter++) 
        coo.spmv_data_race(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s naive COO w/ data race:");

    // naive COO - w/ atomic updates
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        coo.spmv_atomic_updates(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        coo.spmv_atomic_updates(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s naive COO w/ atomic updates:");

    // MKL COO
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        coo.spmv_mkl(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        coo.spmv_mkl(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s MKL COO:");

    coo.release();

    free(x);
    free(y);
}
