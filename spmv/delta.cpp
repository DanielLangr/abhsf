#include <algorithm>
#include <cstdint>
#include <cassert>
#include <iostream>
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

using timer_type = chrono_timer<>;

using element_t = std::tuple<uint32_t, uint32_t, double>;
using elements_t = std::vector<element_t>;

void read_mtx(const std::string& filename, elements_t& elements, matrix_properties& props) 
{
    std::cout << red << "Matrix market reader log..." << reset << std::endl;

    matrix_market_reader<> reader(&std::cout);
    reader.open(filename);
    
    props = reader.props();

    elements.reserve(props.nnz);
    bool warned = false; // check zero elements explicit storage :(
    bool L = false; // check lower/upper triangular parts only for not-unsymmetric matrices
    bool U = false;
    for (uintmax_t k = 0; k < props.nnz; k++) {
        uintmax_t row, col;
        double val_re;
        reader.next_element(&row, &col, &val_re);

        if (row > col)
            L = true;
        if (col > row)
            U = true;

        if (val_re == 0.0) {
            if (warned == false) {
                std::cout << red << "Matrix file contains zero elements." << reset << std::endl;
                warned = true;
            }
        }
        else
            elements.emplace_back(row, col, val_re);
    }

    if (props.symmetry != matrix_symmetry_t::UNSYMMETRIC) {
        if ((L == true) && (U == true))
            throw std::runtime_error("Elements from both L and U parts stored for not-unsymmetric matrix.");

        std::cout << "Matrix symmetric part stored: " << ((L == true) ? "LOWER" : "UPPER") << std::endl;
    }

    std::cout << green << "... [DONE]" << reset << std::endl;
}

int main(int argc, char* argv[])
{
    elements_t elements;
    matrix_properties props;
    read_mtx(argv[1], elements, props);

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

    // sort lexicographically
    std::sort(elements.begin(), elements.end());

    std::cout << "sizeof(MKL_INT): " << cyan << sizeof(MKL_INT) << reset << std::endl;

    // CSR format for Intel MKL
    std::vector<double> a(nnz_stored);
    std::vector<MKL_INT> ia(props.m + 1);
    std::vector<MKL_INT> ja(nnz_stored);

    for (size_t k = 0; k < nnz_stored; k++) {
        a[k] = std::get<2>(elements[k]);
        ja[k] = std::get<1>(elements[k]);
    }

    ia[0] = 0;
    size_t k = 0;
    size_t row = 0;

    while (k < nnz_stored) {
        while ((k < nnz_stored) && (row == std::get<0>(elements[k])))
            k++;

        row++;
        ia[row] = k;
    }

    assert(ia[props.m] == nnz_stored);

    std::vector<double> x(props.n, 1.0);
    std::vector<double> y(props.m);

    assert(props.m == props.n);
    const MKL_INT m = props.m;
    const char transa = 'N';

    static const int warmup_iters = 10;
    static const int n_iters = 100;

    // warm-up
    for (int iter = 0; iter < warmup_iters; iter++) 
        mkl_cspblas_dcsrgemv(&transa, &m, a.data(), ia.data(), ja.data(), x.data(), y.data());

    timer_type timer(timer_type::start_now);

    for (int iter = 0; iter < n_iters; iter++) 
        mkl_cspblas_dcsrgemv(&transa, &m, a.data(), ia.data(), ja.data(), x.data(), y.data());

    timer.stop();

    uintmax_t n_flops = 2 * nnz_all * n_iters;
    double mega_flops = double(n_flops) / 1.0e6 / timer.seconds();
    std::cout << "Measured MFLOP/S Intel MKL: " << yellow << mega_flops << reset << std::endl;

    // naive CSR32:

    // warm-up
    for (int iter = 0; iter < warmup_iters; iter++) {
        std::fill(y.begin(), y.end(), 0.0);

        for (MKL_INT row = 0; row < m; row++) 
            for (MKL_INT k = ia[row]; k < ia[row + 1]; k++)
                y[row] += x[ja[k]] * a[k];

    }

    timer.start();

    for (int iter = 0; iter < n_iters; iter++) {
        std::fill(y.begin(), y.end(), 0.0);

        for (MKL_INT row = 0; row < m; row++) 
            for (MKL_INT k = ia[row]; k < ia[row + 1]; k++)
                y[row] += x[ja[k]] * a[k];

    }

    timer.stop();

    mega_flops = double(n_flops) / 1.0e6 / timer.seconds();
    std::cout << "Measured MFLOP/S naive CSR: " << yellow << mega_flops << reset << std::endl;

    // delta
    auto& da = ja; // alias
    da.clear();

    uint64_t previous = 0;
    for (auto& elem : elements) {
        auto row = std::get<0>(elem);
        auto col = std::get<1>(elem);
        
        assert(row < (1UL << 31));
        assert(col < (1UL << 31));

        uint64_t current = (((uint64_t)row) << 31) + (uint64_t)col;
        uint64_t delta = current - previous;

        assert(delta < (1UL << 32));

        da.emplace_back(delta);
        
        previous = current;
    }

    // warm-up
    for (int iter = 0; iter < warmup_iters; iter++) {
        previous = 0;
        for (size_t k = 0; k < nnz_stored; k++) {
            uint64_t current = previous + (uint64_t)da[k];

            uint64_t row = current >> 31;
            uint64_t col = current & ((1UL << 31) - 1);

            y[row] += a[k] * x[row];

            previous = current;
        }
    }

    timer.start();

    for (int iter = 0; iter < n_iters; iter++) {
        previous = 0;
        for (size_t k = 0; k < nnz_stored; k++) {
            uint64_t current = previous + (uint64_t)da[k];

            uint64_t row = current >> 31;
            uint64_t col = current & ((1UL << 31) - 1);

            y[row] += a[k] * x[row];

            previous = current;
        }
    }

    timer.stop();

    mega_flops = double(n_flops) / 1.0e6 / timer.seconds();
    std::cout << "Measured MFLOP/S delta: " << yellow << mega_flops << reset << std::endl;

    // delta - w/ vectorization:

    uint32_t* da2;
    if (posix_memalign((void**)(&da2), 64, nnz_stored * sizeof(uint32_t)) != 0)
        throw std::runtime_error("posix_memalign() error");

    previous = 0;
    for (size_t k = 0; k < (nnz_stored / 4); k += 4) {
        uint64_t current;

        for (int i = 0; i < 4; i++) {
            uint32_t row = std::get<0>(elements[k + i]);
            uint32_t col = std::get<1>(elements[k + i]);

            assert(row < (1UL << 30));
            assert(col < (1UL << 30));

            current = (((uint64_t)row) << 30) + (uint64_t)col;
            uint64_t delta = current - previous;

            assert(delta < (1UL << 32));
            da2[k + i] = delta;
        }

        previous = current;
    }

#define VARIANT2

#ifdef VARIANT1
    const __m256i x_ptr = _mm256_set1_epi64x((uint64_t)(&x[0]));
    const __m256i y_ptr = _mm256_set1_epi64x((uint64_t)(&y[0]));
#endif
    const __m256i rtemp = _mm256_set1_epi64x((1UL << 30) - 1);

    // warm-up
    for (int iter = 0; iter < warmup_iters; iter++) {
        __m256i previous = _mm256_setzero_si256();
        for (size_t k = 0; k < (nnz_stored / 4); k += 4) {
            __m128i temp = _mm_load_si128((__m128i*)(da2 + k));
            __m256i deltas = _mm256_cvtepu32_epi64(temp);
            __m256i current = _mm256_add_epi64(previous, deltas);
            __m256i rows = _mm256_srli_epi64(current, 30);
            __m256i cols = _mm256_and_si256(current, rtemp);

#ifdef VARIANT0

            uint64_t row = _mm256_extract_epi64(rows, 0);
            uint64_t col = _mm256_extract_epi64(cols, 0);
            y[row] += a[k] * x[row];
            
            row = _mm256_extract_epi64(rows, 1);
            col = _mm256_extract_epi64(cols, 1);
            y[row] += a[k + 1] * x[row];

            row = _mm256_extract_epi64(rows, 2);
            col = _mm256_extract_epi64(cols, 2);
            y[row] += a[k + 2] * x[row];

            row = _mm256_extract_epi64(rows, 3);
            col = _mm256_extract_epi64(cols, 3);
            y[row] += a[k + 3] * x[row];

#elif defined VARIANT1

            rows = _mm256_slli_epi64(rows, 3); // << 3 = * 8 for doubles
            __m256i y_ptrs = _mm256_add_epi64(y_ptr, rows);

            cols = _mm256_slli_epi64(cols, 3); // << 3 = * 8 for doubles
            __m256i x_ptrs = _mm256_add_epi64(x_ptr, cols);

            double* x_ = (double*)_mm256_extract_epi64(x_ptrs, 0);
            double* y_ = (double*)_mm256_extract_epi64(y_ptrs, 0);
            *y_ += a[k] * *x_;

            x_ = (double*)_mm256_extract_epi64(x_ptrs, 1);
            y_ = (double*)_mm256_extract_epi64(y_ptrs, 1);
            *y_ += a[k + 1] * *x_;

            x_ = (double*)_mm256_extract_epi64(x_ptrs, 2);
            y_ = (double*)_mm256_extract_epi64(y_ptrs, 2);
            *y_ += a[k + 2] * *x_;

            x_ = (double*)_mm256_extract_epi64(x_ptrs, 3);
            y_ = (double*)_mm256_extract_epi64(y_ptrs, 3);
            *y_ += a[k + 3] * *x_;

#elif defined VARIANT2

            __m256d x_ = _mm256_i64gather_pd(&x[0], cols, 8);
            __m256d a_ = _mm256_load_pd(&a[k]);
            __m256d res = _mm256_mul_pd(x_, a_);

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

#endif

            previous = _mm256_set1_epi64x(_mm256_extract_epi64(current, 3));
        }
    }

    timer.start();

    for (int iter = 0; iter < n_iters; iter++) {
        __m256i previous = _mm256_setzero_si256();
        for (size_t k = 0; k < (nnz_stored / 4); k += 4) {
            __m128i temp = _mm_load_si128((__m128i*)(da2 + k));
            __m256i deltas = _mm256_cvtepu32_epi64(temp);
            __m256i current = _mm256_add_epi64(previous, deltas);
            __m256i rows = _mm256_srli_epi64(current, 30);
            __m256i cols = _mm256_and_si256(current, rtemp);
            
#ifdef VARIANT0

            uint64_t row = _mm256_extract_epi64(rows, 0);
            uint64_t col = _mm256_extract_epi64(cols, 0);
            y[row] += a[k] * x[row];
            
            row = _mm256_extract_epi64(rows, 1);
            col = _mm256_extract_epi64(cols, 1);
            y[row] += a[k + 1] * x[row];

            row = _mm256_extract_epi64(rows, 2);
            col = _mm256_extract_epi64(cols, 2);
            y[row] += a[k + 2] * x[row];

            row = _mm256_extract_epi64(rows, 3);
            col = _mm256_extract_epi64(cols, 3);
            y[row] += a[k + 3] * x[row];

#elif defined VARIANT1

            rows = _mm256_slli_epi64(rows, 3); // << 3 = * 8 for doubles
            __m256i y_ptrs = _mm256_add_epi64(y_ptr, rows);

            cols = _mm256_slli_epi64(cols, 3); // << 3 = * 8 for doubles
            __m256i x_ptrs = _mm256_add_epi64(x_ptr, cols);

            double* x_ = (double*)_mm256_extract_epi64(x_ptrs, 0);
            double* y_ = (double*)_mm256_extract_epi64(y_ptrs, 0);
            *y_ += a[k] * *x_;

            x_ = (double*)_mm256_extract_epi64(x_ptrs, 1);
            y_ = (double*)_mm256_extract_epi64(y_ptrs, 1);
            *y_ += a[k + 1] * *x_;

            x_ = (double*)_mm256_extract_epi64(x_ptrs, 2);
            y_ = (double*)_mm256_extract_epi64(y_ptrs, 2);
            *y_ += a[k + 2] * *x_;

            x_ = (double*)_mm256_extract_epi64(x_ptrs, 3);
            y_ = (double*)_mm256_extract_epi64(y_ptrs, 3);
            *y_ += a[k + 3] * *x_;

#elif defined VARIANT2

            __m256d x_ = _mm256_i64gather_pd(&x[0], cols, 8);
            __m256d a_ = _mm256_load_pd(&a[k]);
            __m256d res = _mm256_mul_pd(x_, a_);

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

#endif

            previous = _mm256_set1_epi64x(_mm256_extract_epi64(current, 3));
        }
    }

    timer.stop();

    mega_flops = double(n_flops) / 1.0e6 / timer.seconds();
    std::cout << "Measured MFLOP/S vectorized delta: " << yellow << mega_flops << reset << std::endl;

    free(da2);
}
