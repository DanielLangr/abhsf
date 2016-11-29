#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <immintrin.h>

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

static const int warmup_iters = 10;
static const int n_iters = 100;

using timer_type = chrono_timer<>;

using real_type = double;
using index_type = MKL_INT;

static_assert(sizeof(MKL_INT) == 4, "Size of MKL_INT is not 32 bits");

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
            for (MKL_INT row = 0; row < m_; row++) 
                for (MKL_INT k = ia_[row]; k < ia_[row + 1]; k++)
                    y[row] += a_[k] * x[ja_[k]];
        }

        void spmv_mkl(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            static const char transa = 'N';
            static const double alpha = 1.0;
            static const char matdescra[6] = { 'G', ' ', ' ', 'C', ' ', ' ' };
            mkl_dcsrmv(&transa, &m_, &n_, &alpha, matdescra, 
                    a_.data(), ja_.data(), ia_.data(), ia_.data() + 1, x, &alpha, y);
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

class delta_matrix 
{
    public:
        void from_elements(elements_t& elements, const matrix_properties& props)
        {
            std::sort(elements.begin(), elements.end());

            nnz_ = elements.size();
            a_.reserve(nnz_);
            da_.reserve(nnz_);

            uint64_t previous = 0;
            for (const auto& elem : elements) {
                auto row = std::get<0>(elem);
                auto col = std::get<1>(elem);

                assert(row < (1UL << 31));
                assert(col < (1UL << 31));

                uint64_t current = (((uint64_t)row) << 31) + (uint64_t)col;
                uint64_t delta = current - previous;

                assert(delta < (1UL << 32));

                a_.emplace_back(std::get<2>(elem));
                da_.emplace_back((uint32_t)delta);

                previous = current;
            }
        }

        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            uint64_t previous = 0;
            for (size_t k = 0; k < nnz_; k++) {
             // uint64_t current = previous + (uint64_t)da_[k];

             // uint64_t row = current >> 31;
             // uint64_t col = current & ((1UL << 31) - 1);
                uint64_t row = 0;
                uint64_t col = 0;

                y[row] += a_[k] * x[col];

             // previous = current;
            }
        }

        void release()
        {
            a_.clear();
            a_.shrink_to_fit();
            da_.clear();
            da_.shrink_to_fit();
        }

    private:
        uintmax_t nnz_;
        std::vector<real_type> a_;
        std::vector<uint32_t> da_;
};

class delta_vect_matrix 
{
    public:
        delta_vect_matrix() : a_(nullptr), da_(nullptr) { }
        ~delta_vect_matrix() { release(); }

        void from_elements(elements_t& elements, const matrix_properties& props)
        {
            std::sort(elements.begin(), elements.end());

            nnz_ = elements.size();

            posix_memalign((void**)(&a_), 64, nnz_ * sizeof(real_type));
            posix_memalign((void**)(&da_), 64, nnz_ * sizeof(uint32_t));

            uint64_t previous = 0;
            for (size_t k = 0; k < nnz_; k += 4) {
                for (int i = 0; i < 4; i++) {
                    if (k + i < nnz_) {
                        uint32_t row = std::get<0>(elements[k + i]);
                        uint32_t col = std::get<1>(elements[k + i]);

                        assert(row < (1UL << 30));
                        assert(col < (1UL << 30));

                        uint64_t current = (((uint64_t)row) << 30) + (uint64_t)col;
                        uint64_t delta = current - previous;

                        assert(delta < (1UL << 32));
                        a_[k + i] = std::get<2>(elements[k + i]);
                        da_[k + i] = delta;

                        if (i == 3)
                            previous = current;
                    }
                }
            }
        }

        void spmv_0(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            const __m256i rtemp = _mm256_set1_epi64x((1UL << 30) - 1);
            __m256i previous = _mm256_setzero_si256();

            size_t k = 0;
            for (k = 0; k < (nnz_ - 3); k += 4) {
                __m128i temp = _mm_load_si128((__m128i*)(da_ + k));
                __m256i deltas = _mm256_cvtepu32_epi64(temp);
                __m256i current = _mm256_add_epi64(previous, deltas);
                __m256i rows = _mm256_srli_epi64(current, 30);
                __m256i cols = _mm256_and_si256(current, rtemp);

                uint64_t row = _mm256_extract_epi64(rows, 0);
                uint64_t col = _mm256_extract_epi64(cols, 0);
                y[row] += a_[k] * x[row];

                row = _mm256_extract_epi64(rows, 1);
                col = _mm256_extract_epi64(cols, 1);
                y[row] += a_[k + 1] * x[row];

                row = _mm256_extract_epi64(rows, 2);
                col = _mm256_extract_epi64(cols, 2);
                y[row] += a_[k + 2] * x[row];

                row = _mm256_extract_epi64(rows, 3);
                col = _mm256_extract_epi64(cols, 3);
                y[row] += a_[k + 3] * x[row];

                previous = _mm256_set1_epi64x(_mm256_extract_epi64(current, 3));
            }

            uint64_t previous_ = _mm256_extract_epi64(previous, 0);
            for ( ; k < nnz_; k++) {
                uint64_t current = previous_ + (uint64_t)da_[k];

                uint64_t row = current >> 30;
                uint64_t col = current & ((1UL << 30) - 1);

                y[row] += a_[k] * x[row];
            }
        }

        void spmv_1(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            const __m256i x_ptr = _mm256_set1_epi64x((uint64_t)x);
            const __m256i y_ptr = _mm256_set1_epi64x((uint64_t)y);
            const __m256i rtemp = _mm256_set1_epi64x((1UL << 30) - 1);
            __m256i previous = _mm256_setzero_si256();

            size_t k = 0;
            for (k = 0; k < (nnz_ - 3); k += 4) {
                __m128i temp = _mm_load_si128((__m128i*)(da_ + k));
                __m256i deltas = _mm256_cvtepu32_epi64(temp);
                __m256i current = _mm256_add_epi64(previous, deltas);
                __m256i rows = _mm256_srli_epi64(current, 30);
                __m256i cols = _mm256_and_si256(current, rtemp);

                rows = _mm256_slli_epi64(rows, 3); // << 3 = * 8 for doubles
                __m256i y_ptrs = _mm256_add_epi64(y_ptr, rows);

                cols = _mm256_slli_epi64(cols, 3); // << 3 = * 8 for doubles
                __m256i x_ptrs = _mm256_add_epi64(x_ptr, cols);

                double* x_ = (double*)_mm256_extract_epi64(x_ptrs, 0);
                double* y_ = (double*)_mm256_extract_epi64(y_ptrs, 0);
                *y_ += a_[k] * *x_;

                x_ = (double*)_mm256_extract_epi64(x_ptrs, 1);
                y_ = (double*)_mm256_extract_epi64(y_ptrs, 1);
                *y_ += a_[k + 1] * *x_;

                x_ = (double*)_mm256_extract_epi64(x_ptrs, 2);
                y_ = (double*)_mm256_extract_epi64(y_ptrs, 2);
                *y_ += a_[k + 2] * *x_;

                x_ = (double*)_mm256_extract_epi64(x_ptrs, 3);
                y_ = (double*)_mm256_extract_epi64(y_ptrs, 3);
                *y_ += a_[k + 3] * *x_;

                previous = _mm256_set1_epi64x(_mm256_extract_epi64(current, 3));
            }

            uint64_t previous_ = _mm256_extract_epi64(previous, 0);
            for ( ; k < nnz_; k++) {
                uint64_t current = previous_ + (uint64_t)da_[k];

                uint64_t row = current >> 30;
                uint64_t col = current & ((1UL << 30) - 1);

                y[row] += a_[k] * x[row];
            }
        }

        void spmv_2(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            const __m256i rtemp = _mm256_set1_epi64x((1UL << 30) - 1);
            __m256i previous = _mm256_setzero_si256();

            size_t k = 0;
            for (k = 0; k < (nnz_ - 3); k += 4) {
                __m128i temp = _mm_load_si128((__m128i*)(da_ + k));
                __m256i deltas = _mm256_cvtepu32_epi64(temp);
                __m256i current = _mm256_add_epi64(previous, deltas);
                __m256i rows = _mm256_srli_epi64(current, 30);
                __m256i cols = _mm256_and_si256(current, rtemp);

                __m256d x_ = _mm256_i64gather_pd(x, cols, 8);
                __m256d a__ = _mm256_load_pd(a_ + k);
                __m256d res = _mm256_mul_pd(x_, a__);

                __m128d res_lo = _mm256_extractf128_pd(res, 0);
                __m128d res_hi = _mm256_extractf128_pd(res, 1);
                __m128d res_lo_p = _mm_permute_pd(res_lo, 0b00000001);
                __m128d res_hi_p = _mm_permute_pd(res_hi, 0b00000001);

                uint64_t row0 = _mm256_extract_epi64(rows, 0);
                uint64_t row1 = _mm256_extract_epi64(rows, 1);
                uint64_t row2 = _mm256_extract_epi64(rows, 2);
                uint64_t row3 = _mm256_extract_epi64(rows, 3);

                y[row0] += _mm_cvtsd_f64(res_lo);
                y[row1] += _mm_cvtsd_f64(res_lo_p);
                y[row2] += _mm_cvtsd_f64(res_hi);
                y[row3] += _mm_cvtsd_f64(res_hi_p);

                previous = _mm256_set1_epi64x(_mm256_extract_epi64(current, 3));
            }

            uint64_t previous_ = _mm256_extract_epi64(previous, 0);
            for ( ; k < nnz_; k++) {
                uint64_t current = previous_ + (uint64_t)da_[k];

                uint64_t row = current >> 30;
                uint64_t col = current & ((1UL << 30) - 1);

                y[row] += a_[k] * x[row];
            }
        }

        void release()
        {
            free(a_);
            a_ = nullptr;
            free(da_);
            da_ = nullptr;
        }

    private:
        uintmax_t nnz_;
        real_type* RESTRICT a_;
        uint32_t* RESTRICT da_;
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

    uintmax_t nnz_stored = elements.size();
    uintmax_t nnz_all = 0;
    for (auto& elem : elements) {
        nnz_all++;
        if ((props.symmetry != matrix_symmetry_t::UNSYMMETRIC) && (std::get<0>(elem) != std::get<1>(elem)))
            nnz_all++;
    }

    std::cout << "Nonzeros (.mtx):   " << magenta << std::right << std::setw(20) << props.nnz  << reset << std::endl; 
    std::cout << "Nonzeros (stored): " << magenta << std::right << std::setw(20) << nnz_stored << reset << std::endl; 
    std::cout << "Nonzeros (all):    " << magenta << std::right << std::setw(20) << nnz_all    << reset << std::endl; 

    std::cout << "sizeof(MKL_INT): " << cyan << sizeof(MKL_INT) << reset << std::endl;

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

    // naive CSR32
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        csr.spmv(x, y);
    timer_type timer(timer_type::start_now);
    for (int iter = 0; iter < n_iters; iter++) 
        csr.spmv(x, y);
    timer.stop();
    std::cout << "Measured MFLOP/S naive CSR: " << yellow << n_mflops / timer.seconds() << reset << ", result = "
        << green << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl;

    // MKL CSR32
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        csr.spmv_mkl(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        csr.spmv_mkl(x, y);
    timer.stop();
    std::cout << "Measured MFLOP/S MKL CSR: " << yellow << n_mflops / timer.seconds() << reset << ", result = "
        << green << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl;

    csr.release();

    // delta 
    delta_matrix delta;
    delta.from_elements(elements, props);

    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        delta.spmv(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        delta.spmv(x, y);
    timer.stop();
    std::cout << "Measured MFLOP/S delta: " << yellow << n_mflops / timer.seconds() << reset << ", result = "
        << green << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl;

    delta.release();

    // delta with vectorization
    delta_vect_matrix delta_v;
    delta_v.from_elements(elements, props);

    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        delta_v.spmv_0(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        delta_v.spmv_0(x, y);
    timer.stop();
    std::cout << "Measured MFLOP/S delta w/ vect 0: " << yellow << n_mflops / timer.seconds() << reset
        << ", result = " << green << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl;

    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        delta_v.spmv_1(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        delta_v.spmv_1(x, y);
    timer.stop();
    std::cout << "Measured MFLOP/S delta w/ vect 1: " << yellow << n_mflops / timer.seconds() << reset
        << ", result = " << green << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl;

    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        delta_v.spmv_2(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        delta_v.spmv_2(x, y);
    timer.stop();
    std::cout << "Measured MFLOP/S delta w/ vect 2: " << yellow << n_mflops / timer.seconds() << reset
        << ", result = " << green << result(y, y + props.m) / (double)(warmup_iters + n_iters) << reset << std::endl;

    delta_v.release();

    free(x);
    free(y);
}
