#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <immintrin.h>

#ifdef HAVE_MKL
    #include <mkl.h>
#endif

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

#ifdef HAVE_MKL
static_assert(sizeof(MKL_INT) == 4, "Size of MKL_INT is not 32 bits");
using index_type = MKL_INT;
#else
using index_type = int;
#endif

using element_t = std::tuple<index_type, index_type, real_type>;
using elements_t = std::vector<element_t>;

class csr_matrix
{
    public:
        void from_elements(elements_t& elements, const matrix_properties& props)
        {
            std::sort(elements.begin(), elements.end());

            m_ = props.m;
            n_ = props.n;
            const size_t nnz = elements.size();

            a_.resize(nnz);
            ia_.resize(m_ + 1);
            ja_.resize(nnz);

            for (size_t k = 0; k < nnz; k++) {
                a_[k] = std::get<2>(elements[k]);
                ja_[k] = std::get<1>(elements[k]);
            }

            ia_[0] = 0;
            size_t k = 0;
            size_t row = 0;

            while (k < nnz) {
                while ((k < nnz) && (row == std::get<0>(elements[k])))
                    k++;

                row++;
                ia_[row] = k;
            }

            assert(ia_[m_] == nnz);
        }

        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel for
            for (index_type row = 0; row < m_; row++) 
                for (index_type k = ia_[row]; k < ia_[row + 1]; k++)
                    y[row] += a_[k] * x[ja_[k]];
        }


#ifdef HAVE_MKL
        void spmv_mkl(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            static const char transa = 'N';
            static const double alpha = 1.0;
            static const char matdescra[6] = { 'G', ' ', ' ', 'C', ' ', ' ' };
            mkl_dcsrmv(&transa, &m_, &n_, &alpha, matdescra, 
                    a_.data(), ja_.data(), ia_.data(), ia_.data() + 1, x, &alpha, y);
        }
#endif

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
        std::vector<index_type> ia_;
        std::vector<index_type> ja_;
        index_type m_, n_;
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
    elements_t elements;
    matrix_properties props;
    read_mtx_real(argv[1], elements, props);

    const uintmax_t nnz_stored = elements.size();
    uintmax_t nnz_all = 0;
    for (auto& elem : elements) {
        nnz_all++;
        if ((props.symmetry != matrix_symmetry_t::UNSYMMETRIC) && (std::get<0>(elem) != std::get<1>(elem)))
            nnz_all++;
    }

    std::cout << "Nonzeros (.mtx):   " << magenta << std::right << std::setw(20) << props.nnz  << reset << std::endl; 
    std::cout << "Nonzeros (stored): " << magenta << std::right << std::setw(20) << nnz_stored << reset << std::endl; 
    std::cout << "Nonzeros (all):    " << magenta << std::right << std::setw(20) << nnz_all    << reset << std::endl; 

    std::cout << "sizeof(index_type): " << cyan << sizeof(index_type) << reset << std::endl;
    std::cout << "sizeof(real_type):  " << cyan << sizeof(real_type)  << reset << std::endl;

    assert(props.m == props.n);
 // std::vector<real_type> x(props.n, 0.5);
 // std::vector<real_type> y(props.m, 0.0);
    real_type *x, *y;
    posix_memalign((void**)(&x), 64, props.n * sizeof(real_type));
    posix_memalign((void**)(&y), 64, props.m * sizeof(real_type));
    std::fill(x, x + props.n, 0.5);
    std::fill(y, y + props.m, 0.0);

    // verification
    for (const auto& element : elements) 
        y[std::get<0>(element)] += std::get<2>(element) * x[std::get<1>(element)];
    std::cout << "Expected result = " << green << result(y, y + props.n) << reset << std::endl;

    const double n_mflops = (double)(nnz_all * 2 * n_iters) / 1.0e6;

    // CSR
    csr_matrix csr;
    csr.from_elements(elements, props);

    timer_type timer;
/*
    // naive CSR32
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        csr.spmv(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        csr.spmv(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s naive CSR:");
*/
    // MKL CSR32
#ifdef HAVE_MKL
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        csr.spmv_mkl(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        csr.spmv_mkl(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s MKL CSR:");
#endif

    csr.release();

    free(x);
    free(y);
}
