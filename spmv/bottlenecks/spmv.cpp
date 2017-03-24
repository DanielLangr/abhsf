#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <parallel/algorithm>

#include <immintrin.h>

#ifdef HAVE_PAPI
    #include <papi.h>
#endif

#ifdef HAVE_MKL
    #include <mkl.h>
    #include <abhsf/utils/mkl_csr_spmv_helper.h>
#endif

#include <abhsf/utils/colors.h>
#include <abhsf/utils/coo2csr.h>
#include <abhsf/utils/matrix_properties.h>
#include <abhsf/utils/matrix_market_reader.h>
#include <abhsf/utils/restrict.h>
#include <abhsf/utils/synthetic.h>
#include <abhsf/utils/timer.h>
#include <abhsf/utils/thousands_separator.h>

static const int warmup_iters = 4;
static const int n_iters = 40;

using element_t = std::tuple<int, int, double>;
using elements_t = std::vector<element_t>;

template <typename REAL_T, typename ROW_PTR_T, typename COL_IND_T>
class csr_matrix
{
    public:
        void from_elements(elements_t& elements, const matrix_properties& props)
        {
            m_ = props.m;
            n_ = props.n;
            nnz_ = elements.size();
            coo2csr(elements, m_, a_, ia_, ja_);
        }

        void spmv(const REAL_T* RESTRICT x, REAL_T* RESTRICT y)
        {
         // mkl_csr_spmv_helper<REAL_T>::spmv(m_, n_, a_.data(), ia_.data(), ja_.data(), x, y); // MKL version
         // spmv_naive(x, y);
            spmv_naive_1_nnz_per_row(x, y);
        }

        void spmv_naive(const REAL_T* RESTRICT x, REAL_T* RESTRICT y)
        {
            #pragma omp parallel for schedule(static)
            for (long row = 0; row < m_; row++) 
                for (long k = ia_[row]; k < ia_[row + 1]; k++) 
                    y[row] += a_[k] * x[ja_[k]];
        }

        void spmv_naive_1_nnz_per_row(const REAL_T* RESTRICT x, REAL_T* RESTRICT y)
        {
/*
            #pragma omp parallel for schedule(static)
            for (long first_row = 0; first_row < m_; first_row += 8) {
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 0]), _MM_HINT_T2);
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 1]), _MM_HINT_T2);
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 2]), _MM_HINT_T2);
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 3]), _MM_HINT_T2);
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 4]), _MM_HINT_T2);
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 5]), _MM_HINT_T2);
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 6]), _MM_HINT_T2);
                _mm_prefetch((const char*)(x + ja_[first_row + 64 + 7]), _MM_HINT_T2);

                for (long j = 0; j < 8; j++) {
                    long row = first_row + j;
                    long k = row; // ia_[row];
                    y[row] += a_[k] * x[ja_[k]];
                }
            }
*/

            #pragma omp parallel for schedule(static)
            for (long row = 0; row < m_; row++) {
                long k = ia_[row];
                y[row] += a_[k] * x[ja_[k]];
            }
        }

/*
        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel for schedule(static)
            for (index_type row = 0; row < m_; row++) {
                index_type k = ia_[row];
                while (k + 63 < ia_[row + 1]) {
                    for (int i = 0; i < 64; i++)
                        y[row] += x[ja_[k + i]];
                    k += 64;
                }
                while (k < ia_[row + 1]) 
                    y[row] += x[ja_[k++]];
/ *
                for (index_type k = ia_[row]; k < ia_[row + 1]; k++) {
                 // y[row] += a_[k] * x[ja_[k]];
                    y[row] += x[ja_[k]];
                }
* /
            }
        }
*/

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
        std::vector<REAL_T> a_;
        std::vector<ROW_PTR_T> ia_;
        std::vector<COL_IND_T> ja_;
        long m_, n_, nnz_;
};

/*
class coo_matrix
{
    public:
        void from_elements(elements_t& elements, const matrix_properties& props)
        {
         // std::sort(elements.begin(), elements.end());
            __gnu_parallel::sort(elements.begin(), elements.end());

            m_ = props.m;
            n_ = props.n;
            nnz_ = elements.size();

            a_.resize(nnz_);
            ia_.resize(nnz_);
            ja_.resize(nnz_);

            for (size_t k = 0; k < nnz_; k++) {
                a_[k] = std::get<2>(elements[k]);
                ia_[k] = std::get<0>(elements[k]);
                ja_[k] = std::get<1>(elements[k]);
            }
        }

        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel for
            for (size_t k = 0; k < nnz_; k++)
#pragma omp atomic update
                y[ia_[k]] += a_[k] * x[ja_[k]];
        }

        void spmv_clever(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel
            {
                // split work
                long per = nnz_ / omp_get_num_threads();
                long first = omp_get_thread_num() * per;
                long last = first + per - 1;
                if (last > (nnz_ - 1))
                    last = nnz_ - 1;

                long row = ia_[first];
             // real_type y_ = a_[first] * x[ja_[first]];
                real_type y_ = x[ja_[first]];

                long k = first + 1;
                while (k <= last) {
                    if (ia_[k] != row) {
#pragma omp atomic update
                        y[row] += y_;
                        row = ia_[k];
                     // y_ = a_[k] * x[ja_[k]];
                        y_ = x[ja_[k]];
                        k++;
                        break;
                    }
                 // y_ += a_[k] * x[ja_[k]];
                    y_ += x[ja_[k]];
                    k++;
                }
                while (k <= last) {
                    if (ia_[k] != row) {
                        y[row] += y_;
                        y_ = 0.0;
                        row = ia_[k];
                    }
                 // y_ += a_[k] * x[ja_[k]];
                    y_ += x[ja_[k]];
                    k++;
                }
#pragma omp atomic update
                y[row] += y_;
            }
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
     // std::vector<index_type> ia_, ja_;
        std::vector<uint16_t> ia_, ja_;
        index_type m_, n_, nnz_;
};
*/

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
    std::cout.imbue(std::locale(std::locale(), new thousands_separator));

    using real_type = double;
    using timer_type = chrono_timer<>;

    matrix_properties props;
    elements_t elements;

    // matrix type selection:
    if (strcmp(argv[1], "dense") == 0) {
        std::cout << "Matrix: " << yellow << "DENSE" << reset << std::endl;
        synthetise_dense_matrix(props, elements, 20000);
    }
    else if (strcmp(argv[1], "diagonal") == 0) {
        std::cout << "Matrix: " << yellow << "DIAGONAL" << reset << std::endl;
        synthetise_diagonal_matrix(props, elements, 200000000);
    }
    else if (strcmp(argv[1], "single-column") == 0) {
        std::cout << "Matrix: " << yellow << "SINGLE-COLUMN" << reset << std::endl;
        synthetise_single_column_matrix(props, elements, 200000000);
    }
    else if (strcmp(argv[1], "random-column") == 0) {
        std::cout << "Matrix: " << yellow << "RANDOM-COLUMN" << reset << std::endl;
        synthetise_random_column_matrix(props, elements, 200000000);
    }
    else if (strcmp(argv[1], "single-row-column") == 0) {
        std::cout << "Matrix: " << yellow << "SINGLE-ROW-COLUMN" << reset << std::endl;
        synthetise_single_row_column_matrix(props, elements, 100000000);
    }
    else if (strcmp(argv[1], "single-row") == 0) {
        std::cout << "Matrix: " << yellow << "SINGLE-ROW" << reset << std::endl;
        synthetise_single_row_matrix(props, elements, 200000000);
    }
    else // read from file if not synthetic
        read_mtx_real_or_binary(argv[1], elements, props);

    // print out final dimensions
    std::cout << "Matrix dimensions: " << magenta << 
        props.m << " x " << props.n << " : " << props.nnz << " = " << elements.size() << reset << std::endl;

    // check matrix:
    if (props.symmetry != matrix_symmetry_t::UNSYMMETRIC)
        throw std::runtime_error("This program does support unsymmetric matrices only!");
    if (props.m != props.n)
        throw std::runtime_error("This program does support square matrices only!");

    // vectors:
    static const double x_element = 0.5;
    real_type *x, *y;
    posix_memalign((void**)(&x), 64, props.n * sizeof(real_type));
    posix_memalign((void**)(&y), 64, props.m * sizeof(real_type));
    std::fill(x, x + props.n, x_element);
    std::fill(y, y + props.m, 0.0);

    // verification
    std::vector<double> y_temp(props.m, 0.0);
    for (const auto& element : elements) 
        y_temp[std::get<0>(element)] += std::get<2>(element) * x_element;
    std::cout << "Expected result = " << green << result(y_temp.cbegin(), y_temp.cend()) << reset << std::endl;
    y_temp.clear();
    y_temp.shrink_to_fit();

    const long nnz = elements.size();
    const double n_mflops = (double)(nnz * 2 * n_iters) / 1.0e6;

 // csr_matrix<real_type, MKL_INT, MKL_INT> A; 
    csr_matrix<real_type, int, int> A;
 // coo_matrix A;

    A.from_elements(elements, props);
    timer_type timer;

    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        A.spmv(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters - 1; iter++) 
        A.spmv(x, y);
    // last iteration:
#ifdef HAVE_PAPI
    int papi_events[4] = { PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM, PAPI_TLB_DM };
    long long papi_values[4];
    PAPI_start_counters(papi_events, 4);
#endif
    A.spmv(x, y);
#ifdef HAVE_PAPI
    PAPI_stop_counters(papi_values, 4);
    std::cout << "L1 cache misses: "
        << magenta << std::right << std::setw(20) << papi_values[0] << reset << std::endl;
    std::cout << "L2 cache misses: "
        << magenta << std::right << std::setw(20) << papi_values[1] << reset << std::endl;
    std::cout << "L3 cache misses: "
        << magenta << std::right << std::setw(20) << papi_values[2] << reset << std::endl;
    std::cout << "TLB misses:      "
        << magenta << std::right << std::setw(20) << papi_values[3] << reset << std::endl;
#endif


    timer.stop();

    // result
    std::cout << std::left << std::setw(24) << "Measured MFLOP/s:"
        << yellow << std::right << std::fixed << std::setprecision(1) << std::setw(10) 
        << n_mflops / timer.seconds() << reset << ", result = " << green << std::setprecision(3) 
        << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl; 

    A.release();

    free(x);
    free(y);
}
