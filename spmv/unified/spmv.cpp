#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <parallel/algorithm>

#include <immintrin.h>

#ifdef HAVE_MKL
    #include <mkl.h>
#endif

#ifdef HAVE_CUDA
    #include <cuda_runtime.h>
    #include <cusparse.h>
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

#define DELTA_K(K) \
    do { \
        if ((props.n > (1UL << (K - 1))) && (props.n <= (1UL << K))) \
            delta_spmv_benchmark<K>(elements, props, x, y); \
    } while (0)

static const int warmup_iters = 5;
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
         // std::sort(elements.begin(), elements.end());
            __gnu_parallel::sort(elements.begin(), elements.end());

            m_ = props.m;
            n_ = props.n;
            nnz_ = elements.size();

            a_.resize(nnz_);
            ia_.resize(m_ + 1);
            ja_.resize(nnz_);

            for (size_t k = 0; k < nnz_; k++) {
                a_[k] = std::get<2>(elements[k]);
                ja_[k] = std::get<1>(elements[k]);
            }

            ia_[0] = 0;
            size_t k = 0;
            size_t row = 0;

            while (k < nnz_) {
                while ((k < nnz_) && (row == std::get<0>(elements[k])))
                    k++;

                row++;
                ia_[row] = k;
            }

            assert(ia_[m_] == nnz_);

#ifdef HAVE_CUDA
            static_assert(sizeof(index_type) == sizeof(int), "Indexing type must be int");

            if (cudaMalloc((void**)&ca_, nnz_ * sizeof(real_type)) != cudaSuccess)
                throw std::runtime_error("Error running cudaMalloc function!");
            if (cudaMalloc((void**)&cia_, (m_ + 1) * sizeof(index_type)) != cudaSuccess)
                throw std::runtime_error("Error running cudaMalloc function!");
            if (cudaMalloc((void**)&cja_, nnz_ * sizeof(index_type)) != cudaSuccess)
                throw std::runtime_error("Error running cudaMalloc function!");
            if (cudaMemcpy(ca_, a_.data(), nnz_ * sizeof(real_type), cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("Error running cudaMemcpy function!");
            if (cudaMemcpy(cia_, ia_.data(), (m_ + 1) * sizeof(index_type), cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("Error running cudaMemcpy function!");
            if (cudaMemcpy(cja_, ja_.data(), nnz_ * sizeof(index_type), cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("Error running cudaMemcpy function!");

            if (cusparseCreateMatDescr(&cdescr_) != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error("Error running cusparseCreateMatDescr function!");
            cusparseSetMatType(cdescr_, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(cdescr_, CUSPARSE_INDEX_BASE_ZERO);
#endif
        }

#ifdef HAVE_CUDA
        void to_hyb (cusparseHandle_t handle)
        {
            if (cusparseCreateHybMat(&chybmat_) != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error("Error running cusparseCreateHybMat function!");
            if (cusparseDcsr2hyb(handle, m_, n_, cdescr_,
                        ca_, cia_, cja_, chybmat_, 0, CUSPARSE_HYB_PARTITION_AUTO) != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error("Error running cusparseDcsr2hyb function!");
        }
#endif

        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel for
            for (index_type row = 0; row < m_; row++) 
                for (index_type k = ia_[row]; k < ia_[row + 1]; k++)
                 // y[row] += a_[k] * x[ja_[k]];
                    y[row] += x[ja_[k]];
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

#ifdef HAVE_CUDA
        void spmv_cusparse(const real_type* x, real_type* y, cusparseHandle_t handle)
        {
            static const double alpha = 1.0;
            if (cusparseDcsrmv(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_, n_, nnz_, &alpha, cdescr_,
                    ca_, cia_, cja_, x, &alpha, y) != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error("Error running cusparseDcsrmv function!");
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("Error running cudaDeviceSynchronize function!");
        }
/*
        // CUDA 8.0 and later
        void spmv_cusparse_mp(const real_type* x, real_type* y, cusparseHandle_t handle)
        {
            static const double alpha = 1.0;
            if (cusparseDcsrmv_mp(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_, n_, nnz_, &alpha, cdescr_,
                    ca_, cia_, cja_, x, &alpha, y) != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error("Error running cusparseDcsrmv function!");
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("Error running cudaDeviceSynchronize function!");
        }
*/

        void spmv_cusparse_hyb(const real_type* x, real_type* y, cusparseHandle_t handle)
        {
            static const double alpha = 1.0;
            if (cusparseDhybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, cdescr_, chybmat_, x, &alpha, y) != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error("Error running cusparseDhybmv function!");
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("Error running cudaDeviceSynchronize function!");
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

#ifdef HAVE_CUDA
            cusparseDestroyHybMat(chybmat_);
            cusparseDestroyMatDescr(cdescr_);
            if (ca_) cudaFree(ca_);
            if (cia_) cudaFree(cia_);
            if (cja_) cudaFree(cja_);
#endif
        }

    private:
        std::vector<real_type> a_;
        std::vector<index_type> ia_;
        std::vector<index_type> ja_;
        index_type m_, n_, nnz_;

#ifdef HAVE_CUDA
        real_type* ca_;
        index_type* cia_;
        index_type* cja_;
        cusparseMatDescr_t cdescr_;
        cusparseHybMat_t chybmat_;
#endif
};

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
                real_type y_ = a_[first] * x[ja_[first]];

                long k = first + 1;
                while (k <= last) {
                    if (ia_[k] != row) {
#pragma omp atomic update
                        y[row] += y_;
                        row = ia_[k];
                        y_ = a_[k] * x[ja_[k]];
                        k++;
                        break;
                    }
                    y_ += a_[k] * x[ja_[k]];
                    k++;
                }
                while (k <= last) {
                    if (ia_[k] != row) {
                        y[row] += y_;
                        y_ = 0.0;
                        row = ia_[k];
                    }
                    y_ += a_[k] * x[ja_[k]];
                    k++;
                }
#pragma omp atomic update
                y[row] += y_;
            }
        }

/*
        void spmv_map(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel
            {
                std::map<index_type, real_type> map;
           
#pragma omp for
                for (size_t k = 0; k < nnz_; k++) {
                    const index_type row = ia_[k];

                    auto iter = map.find(row);
                    if ((iter == map.end()) && (map.size() >= 16)) {
                        for (auto jter = map.cbegin(); jter != map.cend(); jter++)
#pragma omp atomic update
                            y[jter->first] += jter->second;
                        map.clear();
                    }
                    map[row] += a_[k] * x[ja_[k]];
                }
            }
        }
*/

#ifdef HAVE_MKL
        void spmv_mkl(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            static const char transa = 'N';
            static const double alpha = 1.0;
            static const char matdescra[6] = { 'G', ' ', ' ', 'C', ' ', ' ' };
            mkl_dcoomv(&transa, &m_, &n_, &alpha, matdescra, 
                    a_.data(), ia_.data(), ja_.data(), &nnz_, x, &alpha, y);
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
        index_type m_, n_, nnz_;
};

/*
class delta_matrix
{
    public:
        delta_matrix() : a_(nullptr), da_(nullptr) { }
        ~delta_matrix() { release(); }

        void from_elements(elements_t& elements, const matrix_properties& props)
        {
         // std::sort(elements.begin(), elements.end());
            __gnu_parallel::sort(elements.begin(), elements.end());

            m_ = props.m;
            n_ = props.n;
            nnz_ = elements.size();

            posix_memalign((void**)(&a_), 64, nnz_ * sizeof(real_type));
            posix_memalign((void**)(&da_), 64, nnz_ * sizeof(uint32_t));

#pragma omp parallel
            {
#pragma omp single
                {
                    thread_first_index.resize(omp_get_num_threads());
                    thread_last_index.resize(omp_get_num_threads());
                    thread_first_current.resize(omp_get_num_threads());
                }

                uint64_t T = (uint64_t)omp_get_num_threads();
                uint64_t t = (uint64_t)omp_get_thread_num();

                uint64_t per = nnz_ / T;
                uint64_t first = t * per;
                uint64_t last = first + per - 1;
                if (last > (nnz_ - 1))
                    last = nnz_ - 1;

                thread_first_index[t] = first;
                thread_last_index[t] = last;

                uint64_t row = std::get<0>(elements[first]);
                uint64_t col = std::get<1>(elements[first]);
                uint64_t current = (row << 31) + col;
                thread_first_current[t] = current;

                uint64_t previous = current;
                for (size_t k = first; k <= last; k++) {
                    auto row = std::get<0>(elements[k]);
                    auto col = std::get<1>(elements[k]);

                    assert(row < (1UL << 31));
                    assert(col < (1UL << 31));

                    uint64_t current = (((uint64_t)row) << 31) + (uint64_t)col;
                    uint64_t delta = current - previous;

                    assert(delta < (1UL << 32));

                    a_[k] = std::get<2>(elements[k]);
                    da_[k] = (uint32_t)delta;

                    previous = current;
                }
            }
        }

        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel
            {
                uint64_t t = (uint64_t)omp_get_thread_num();
                uint64_t first = thread_first_index[t];
                uint64_t last = thread_last_index[t];
                uint64_t previous = thread_first_current[t];

                uint64_t row = previous >> 31; // first thread row
                uint64_t col = previous & ((1UL << 31) - 1);
             // real_type y_ = a_[first] * x[col];
                real_type y_ = x[col];

                uint64_t k = first + 1;
                while (k <= last) {
                    uint64_t current = previous + (uint64_t)da_[k];
                    previous = current;

                    uint64_t row_ = current >> 31;
                    col = current & ((1UL << 31) - 1);

                    if (row_ != row) {
#pragma omp atomic update
                        y[row] += y_;
                        row = row_;
                     // y_ = a_[k] * x[col];
                        y_ = x[col];
                        k++;
                        break;
                    }
                 // y_ += a_[k] * x[col];
                    y_ += x[col];
                    k++;
                }
                while (k <= last) {
                    uint64_t current = previous + (uint64_t)da_[k];
                    previous = current;

                    uint64_t row_ = current >> 31;
                    col = current & ((1UL << 31) - 1);

                    if (row_ != row) {
                        y[row] += y_;
                        y_ = 0.0;
                        row = row_;
                    }
                 // y_ += a_[k] * x[col];
                    y_ += x[col];
                    k++;
                }
#pragma omp atomic update
                y[row] += y_;
            }
        }

        void release()
        {
            free(a_);
            a_ = nullptr;
            free(da_);
            da_ = nullptr;

            thread_first_index.clear();
            thread_first_index.shrink_to_fit();
            thread_last_index.clear();
            thread_last_index.shrink_to_fit();
            thread_first_current.clear();
            thread_first_current.shrink_to_fit();
        }

    private:
        uint64_t m_, n_, nnz_;
        real_type* RESTRICT a_;
        uint32_t* RESTRICT da_;
        std::vector<uint64_t> thread_first_index;
        std::vector<uint64_t> thread_last_index;
        std::vector<uint64_t> thread_first_current;
};
*/

// cache-blocking version:
class delta_matrix
{
    public:
        delta_matrix() : a_(nullptr), da_(nullptr) { }
        ~delta_matrix() { release(); }

        void from_elements(elements_t& elements, const matrix_properties& props)
        {
         // std::sort(elements.begin(), elements.end());
            __gnu_parallel::sort(elements.begin(), elements.end(),
                    [](const element_t& a, const element_t& b) {
                        long row_a = std::get<0>(a); long col_a = std::get<1>(a);
                        long row_b = std::get<0>(b); long col_b = std::get<1>(b);

                        assert(sizeof(real_type) == 8);
                        static const long epcl = 64 / sizeof(real_type); // elements per cache line

                        long block_row_a = row_a / epcl; long block_col_a = col_a / epcl;
                        long block_row_b = row_b / epcl; long block_col_b = col_b / epcl;

                        long local_row_a = row_a % epcl; long local_col_a = col_a % epcl;
                        long local_row_b = row_b % epcl; long local_col_b = col_b % epcl;

                        return (std::make_tuple(block_row_a, block_col_a, local_row_a, local_col_a)
                                < std::make_tuple(block_row_b, block_col_b, local_row_b, local_col_b));
                    });

            m_ = props.m;
            n_ = props.n;
            nnz_ = elements.size();

            posix_memalign((void**)(&a_), 64, nnz_ * sizeof(real_type));
            posix_memalign((void**)(&da_), 64, nnz_ * sizeof(uint32_t));

#pragma omp parallel
            {
#pragma omp single
                {
                    thread_first_index.resize(omp_get_num_threads());
                    thread_last_index.resize(omp_get_num_threads());
                    thread_first_current.resize(omp_get_num_threads());
                }

                uint64_t T = (uint64_t)omp_get_num_threads();
                uint64_t t = (uint64_t)omp_get_thread_num();

                uint64_t per = nnz_ / T;
                uint64_t first = t * per;
                uint64_t last = first + per - 1;
                if (last > (nnz_ - 1))
                    last = nnz_ - 1;

                thread_first_index[t] = first;
                thread_last_index[t] = last;

                uint64_t row = std::get<0>(elements[first]);
                uint64_t col = std::get<1>(elements[first]);
                uint64_t current = ((row / 8) << (28 + 3 + 3))
                    + ((col / 8) << (3 + 3)) + ((row % 8) << 3) + (col % 8);
                thread_first_current[t] = current;

                uint64_t previous = current;
                for (size_t k = first; k <= last; k++) {
                    uint64_t row = std::get<0>(elements[k]);
                    uint64_t col = std::get<1>(elements[k]);

                    assert(row < (1UL << 31));
                    assert(col < (1UL << 31));

                    uint64_t current = ((row / 8) << (28 + 3 + 3))
                        + ((col / 8) << (3 + 3)) + ((row % 8) << 3) + (col % 8);
                    uint64_t delta = current - previous;

                    assert(delta < (1UL << 32));

                    a_[k] = std::get<2>(elements[k]);
                    da_[k] = (uint32_t)delta;

                    previous = current;
                }
            }
        }

        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel
            {
                uint64_t t = (uint64_t)omp_get_thread_num();
                uint64_t first = thread_first_index[t];
                uint64_t last = thread_last_index[t];
                uint64_t previous = thread_first_current[t];

                // first thread block row
                uint64_t block_row = previous >> (28 + 3 + 3);
                uint64_t block_col = (previous >> (3 + 3)) & ((1UL << 28) - 1);
                uint64_t local_row = (previous >> 3) & 7;
                uint64_t local_col = previous & 7;
                real_type* RESTRICT y_ptr = y + (block_row << 3);
                const real_type* x_ptr = x + (block_col << 3);

                real_type y_[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
                y_[local_row] = /* a_[first] * */ x_ptr[local_col];

                uint64_t k = first + 1;

                while (k <= last) {
                    uint64_t current = previous + (uint64_t)da_[k];
                    previous = current;

                    uint64_t block_row_ = current >> (28 + 3 + 3);
                    block_col = (current >> (3 + 3)) & ((1UL << 28) - 1);
                    local_row = (current >> 3) & 7;
                    local_col = current & 7;
                    x_ptr = x + (block_col << 3);

                    if (block_row_ != block_row) {
                        for (int i = 0; i < 7; i++) {
#pragma omp atomic update
                            y_ptr[i] += y_[i];
                        }
                        block_row = block_row_;
                        y_ptr = y + (block_row << 3);
                        for (int i = 0; i < 7; i++) y_[i] = 0.0;

                        y_[local_row] = /* a_[k] * */ x_ptr[local_col];
                        k++;
                        break;
                    }
                    y_[local_row] = /* a_[k] * */ x_ptr[local_col];
                    k++;
                }

                while (k <= last) {
                    uint64_t current = previous + (uint64_t)da_[k];
                    previous = current;

                    uint64_t block_row_ = current >> (28 + 3 + 3);
                    block_col = (current >> (3 + 3)) & ((1UL << 28) - 1);
                    local_row = (current >> 3) & 7;
                    local_col = current & 7;
                    x_ptr = x + (block_col << 3);

                    if (block_row_ != block_row) {
                        for (int i = 0; i < 7; i++) {
#pragma omp atomic update
                            y_ptr[i] += y_[i];
                        }
                        block_row = block_row_;
                        y_ptr = y + (block_row << 3);
                        for (int i = 0; i < 7; i++) y_[i] = 0.0;
                    }
                    y_[local_row] = /* a_[k] * */ x_ptr[local_col];
                    k++;
                }
                for (int i = 0; i < 7; i++) {
#pragma omp atomic update
                    y_ptr[i] += y_[i];
                }
            }
        }

        void release()
        {
            free(a_);
            a_ = nullptr;
            free(da_);
            da_ = nullptr;

            thread_first_index.clear();
            thread_first_index.shrink_to_fit();
            thread_last_index.clear();
            thread_last_index.shrink_to_fit();
            thread_first_current.clear();
            thread_first_current.shrink_to_fit();
        }

    private:
        uint64_t m_, n_, nnz_;
        real_type* RESTRICT a_;
        uint32_t* RESTRICT da_;
        std::vector<uint64_t> thread_first_index;
        std::vector<uint64_t> thread_last_index;
        std::vector<uint64_t> thread_first_current;
};

/*
template <int K>
class delta_matrix
{
    public:
        delta_matrix() : a_(nullptr), da_(nullptr) { }
        ~delta_matrix() { release(); }

        void from_elements(elements_t& elements, const matrix_properties& props)
        {
         // std::sort(elements.begin(), elements.end());
            __gnu_parallel::sort(elements.begin(), elements.end());

            m_ = props.m;
            n_ = props.n;
            nnz_ = elements.size();

            posix_memalign((void**)(&a_), 64, nnz_ * sizeof(real_type));
         // posix_memalign((void**)(&da_), 64, nnz_ * sizeof(uint32_t));
            posix_memalign((void**)(&da_), 64, nnz_ * 8); // worst-case; TODO: calculate real space required

#pragma omp parallel
            {
#pragma omp single
                {
                    thread_first_index.resize(omp_get_num_threads());
                    thread_last_index.resize(omp_get_num_threads());
                    thread_first_current.resize(omp_get_num_threads());
                }

                uint64_t T = (uint64_t)omp_get_num_threads();
                uint64_t t = (uint64_t)omp_get_thread_num();

                uint64_t per = nnz_ / T;
                uint64_t first = t * per;
                uint64_t last = first + per - 1;
                if (last > (nnz_ - 1))
                    last = nnz_ - 1;

                thread_first_index[t] = first;
                thread_last_index[t] = last;

                uint64_t row = std::get<0>(elements[first]);
                uint64_t col = std::get<1>(elements[first]);
                uint64_t current = (row << K) + col;
                thread_first_current[t] = current;

                uint64_t previous = current;
                uint8_t* addr = da_ + first * 8;
                for (size_t k = first + 1; k <= last; k++) {
                    auto row = std::get<0>(elements[k]);
                    auto col = std::get<1>(elements[k]);

                    assert(row < (1UL << K));
                    assert(col < (1UL << K));

                    uint64_t current = (((uint64_t)row) << K) + (uint64_t)col;
                    uint64_t delta = current - previous;

                    if (delta <  (1UL << 6)) {
                        uint8_t delta6 = (uint8_t)delta << 2;
                        *addr = delta6;
                        addr += 1;
                    }
                    else if (delta < (1UL << 14)) {
                        uint16_t delta14 = (uint16_t)delta << 2;
                        delta14 |= 1;
                        *((uint16_t*)addr) = delta14;
                        addr += 2;
                    }
                    else if (delta < (1UL << 30)) {
                        uint32_t delta30 = (uint32_t)delta << 2;
                        delta30 |= 2;
                        *((uint32_t*)addr) = delta30;
                        addr += 4;
                    }
                    else {
                        uint64_t delta62 = delta << 2;
                        delta62 |= 3;
                        *((uint64_t*)addr) = delta62;
                        addr += 8;
                    }

                    a_[k] = std::get<2>(elements[k]);
                 // da_[k] = (uint32_t)delta;

                    previous = current;
                }
            }
        }

        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel
            {
                uint64_t t = (uint64_t)omp_get_thread_num();
                uint64_t first = thread_first_index[t];
                uint64_t last = thread_last_index[t];
                uint64_t previous = thread_first_current[t];

                uint64_t row = previous >> K; // first thread row
                uint64_t col = previous & ((1UL << K) - 1);
             // real_type y_ = a_[first] * x[col];
                real_type y_ = x[col];

                uint64_t k = first + 1;
                uint8_t* addr = da_ + first * 8;
                while (k <= last) {
                    uint8_t tag = *addr & 3;
                    uint64_t delta;
                    if (tag == 0) {
                        uint8_t delta6 = *addr;
                        delta = (uint64_t)(delta6 >> 2);
                        addr += 1;
                    }
                    else if (tag == 1) {
                        uint16_t delta14 = *((uint16_t*)addr);
                        delta = (uint64_t)(delta14 >> 2);
                        addr += 2;
                    }
                    else if (tag == 2) {
                        uint32_t delta30 = *((uint32_t*)addr);
                        delta = (uint64_t)(delta30 >> 2);
                        addr += 4;
                    }
                    else {
                        uint64_t delta62 = *((uint64_t*)addr);
                        delta = delta62 >> 2;
                        addr += 8;
                    }

                    uint64_t current = previous + delta;
                    previous = current;

                    uint64_t row_ = current >> K;
                    col = current & ((1UL << K) - 1);

                    if (row_ != row) {
#pragma omp atomic update
                        y[row] += y_;
                        row = row_;
                     // y_ = a_[k] * x[col];
                        y_ = x[col];
                        k++;
                        break;
                    }
                 // y_ += a_[k] * x[col];
                    y_ += x[col];
                    k++;
                }
                while (k <= last) {
                    uint8_t tag = *addr & 3;
                    uint64_t delta;
                    if (tag == 0) {
                        uint8_t delta6 = *addr;
                        delta = (uint64_t)(delta6 >> 2);
                        addr += 1;
                    }
                    else if (tag == 1) {
                        uint16_t delta14 = *((uint16_t*)addr);
                        delta = (uint64_t)(delta14 >> 2);
                        addr += 2;
                    }
                    else if (tag == 2) {
                        uint32_t delta30 = *((uint32_t*)addr);
                        delta = (uint64_t)(delta30 >> 2);
                        addr += 4;
                    }
                    else {
                        uint64_t delta62 = *((uint64_t*)addr);
                        delta = delta62 >> 2;
                        addr += 8;
                    }

                    uint64_t current = previous + delta;
                    previous = current;

                    uint64_t row_ = current >> K;
                    col = current & ((1UL << K) - 1);

                    if (row_ != row) {
                        y[row] += y_;
                        y_ = 0.0;
                        row = row_;
                    }
                 // y_ += a_[k] * x[col];
                    y_ += x[col];
                    k++;
                }
#pragma omp atomic update
                y[row] += y_;
            }
        }

        void release()
        {
            free(a_);
            a_ = nullptr;
            free(da_);
            da_ = nullptr;

            thread_first_index.clear();
            thread_first_index.shrink_to_fit();
            thread_last_index.clear();
            thread_last_index.shrink_to_fit();
            thread_first_current.clear();
            thread_first_current.shrink_to_fit();
        }

    private:
        uint64_t m_, n_, nnz_;
        real_type* RESTRICT a_;
     // uint32_t* RESTRICT da_;
        uint8_t* RESTRICT da_;
        std::vector<uint64_t> thread_first_index;
        std::vector<uint64_t> thread_last_index;
        std::vector<uint64_t> thread_first_current;
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
/*
template <int K>
void delta_spmv_benchmark(
        elements_t& elements, const matrix_properties& props,
        const real_type* RESTRICT x, real_type* RESTRICT y)
{
    std::cout << "Delta K exponent: " << cyan << K << reset << std::endl;

    uintmax_t nnz_all = elements.size();
    const double n_mflops = (double)(nnz_all * 2 * n_iters) / 1.0e6;

    delta_matrix<K> A;
    A.from_elements(elements, props);

    timer_type timer;

    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        A.spmv(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        A.spmv(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s:");

    A.release();
}
*/
int main(int argc, char* argv[])
{
    elements_t elements;
    matrix_properties props;

    // matrix type selection
    if (strcmp(argv[1], "dense") == 0) {
        std::cout << "Matrix: " << yellow << "DENSE" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 20000;
        props.nnz = props.m * props.n;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            for (size_t j = 0; j < props.n; j++)
                elements.emplace_back(i, j, 1.0);
    }
    else if (strcmp(argv[1], "diagonal") == 0) {
        std::cout << "Matrix: " << yellow << "DIAGONAL" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 200000000L;
        props.nnz = props.m;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            elements.emplace_back(i, i, 1.0);
    }
    else if (strcmp(argv[1], "single-column") == 0) {
        std::cout << "Matrix: " << yellow << "SINGLE-COLUMN" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 200000000L;
        props.nnz = props.m;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            elements.emplace_back(i, 0, 1.0);
    }
    else if (strcmp(argv[1], "random-column") == 0) {
        std::cout << "Matrix: " << yellow << "RANDOM-COLUMN" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 200000000L;
        props.nnz = props.m;

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, props.n - 1);

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.m; i++)
            elements.emplace_back(i, dist(mt), 1.0);
    }
    else if (strcmp(argv[1], "single-row-column") == 0) {
        std::cout << "Matrix: " << yellow << "SINGLE-ROW-COLUMN" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 100000000L;
        props.nnz = props.m + props.n - 1;

        elements.reserve(props.nnz);
        for (size_t j = 0; j < props.n; j++)
            elements.emplace_back(0, j, 1.0);
        for (size_t i = 1; i < props.m; i++)
            elements.emplace_back(i, 0, 1.0);
    }
    else if (strcmp(argv[1], "single-row") == 0) {
        std::cout << "Matrix: " << yellow << "SINGLE-ROW" << reset << std::endl;

        props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
        props.type = matrix_type_t::REAL;
        props.m = props.n = 200000000L;
        props.nnz = props.n;

        elements.reserve(props.nnz);
        for (size_t i = 0; i < props.n; i++)
            elements.emplace_back(props.m - 1, i, 1.0);
    }
    else 
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
    static const double x_element = 0.5;
 // std::vector<real_type> x(props.n, 0.5);
 // std::vector<real_type> y(props.m, 0.0);
    real_type *x, *y;
    posix_memalign((void**)(&x), 64, props.n * sizeof(real_type));
    posix_memalign((void**)(&y), 64, props.m * sizeof(real_type) + 8); // ensure whole cache final cache line
    std::fill(x, x + props.n, x_element);
    std::fill(y, y + props.m, 0.0);

#ifdef HAVE_CUDA
    real_type *cx, *cy;
    if (cudaMalloc((void**)&cx, props.n * sizeof(real_type)) != cudaSuccess)
        throw std::runtime_error("Error running cudaMalloc function!");
    if (cudaMalloc((void**)&cy, props.m * sizeof(real_type)) != cudaSuccess)
        throw std::runtime_error("Error running cudaMalloc function!");
    if (cudaMemcpy(cx, x, props.n * sizeof(real_type), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Error running cudaMemcpy function!");
#endif

    // verification
    std::vector<double> y_temp(props.m, 0.0);
    for (const auto& element : elements) 
        y_temp[std::get<0>(element)] += std::get<2>(element) * x_element;
    std::cout << "Expected result = " << green << result(y_temp.cbegin(), y_temp.cend()) << reset << std::endl;
    y_temp.clear();
    y_temp.shrink_to_fit();

    const double n_mflops = (double)(nnz_all * 2 * n_iters) / 1.0e6;

 // csr_matrix A;
 // coo_matrix A;
    delta_matrix A;
/*
    DELTA_K(1); DELTA_K(2); DELTA_K(3); DELTA_K(4); DELTA_K(5); DELTA_K(6); DELTA_K(7); DELTA_K(8);
    DELTA_K(9); DELTA_K(10); DELTA_K(11); DELTA_K(12); DELTA_K(13); DELTA_K(14); DELTA_K(15); DELTA_K(16);
    DELTA_K(17); DELTA_K(18); DELTA_K(19); DELTA_K(20); DELTA_K(21); DELTA_K(22); DELTA_K(23); DELTA_K(24);
    DELTA_K(25); DELTA_K(26); DELTA_K(27); DELTA_K(28); DELTA_K(29); DELTA_K(30); DELTA_K(31); DELTA_K(32);
    DELTA_K(33); DELTA_K(34); DELTA_K(35); DELTA_K(36); DELTA_K(37); DELTA_K(38); DELTA_K(39); DELTA_K(40);
    DELTA_K(41); DELTA_K(42); DELTA_K(43); DELTA_K(44); DELTA_K(45); DELTA_K(46); DELTA_K(47); DELTA_K(48);
    DELTA_K(49); DELTA_K(50); DELTA_K(51); DELTA_K(52); DELTA_K(53); DELTA_K(54); DELTA_K(55); DELTA_K(56);
    DELTA_K(57); DELTA_K(58); DELTA_K(59); DELTA_K(60); DELTA_K(61); DELTA_K(62); DELTA_K(63); // DELTA_K(64);
*/

    A.from_elements(elements, props);
    timer_type timer;

    // naive 
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
     // A.spmv_clever(x, y);
        A.spmv(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
     // A.spmv_clever(x, y);
        A.spmv(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s:");

/*
    // MKL 
#ifdef HAVE_MKL
    std::fill(y, y + props.m, 0.0);
    for (int iter = 0; iter < warmup_iters; iter++) 
        A.spmv_mkl(x, y);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
        A.spmv_mkl(x, y);
    timer.stop();
    PRINT_RESULT("Measured MFLOP/s MKL:");
#endif
*/
#ifdef HAVE_CUDA
    cusparseHandle_t handle;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("Error running cusparseCreate function!");

    A.to_hyb(handle);

    std::fill(y, y + props.m, 0.0);
    if (cudaMemcpy(cy, y, props.m * sizeof(real_type), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Error running cudaMemcpy function!");
    for (int iter = 0; iter < warmup_iters; iter++) 
     // A.spmv_cusparse_mp(cx, cy, handle); // available since CUDA 8.0
     // A.spmv_cusparse(cx, cy, handle);
        A.spmv_cusparse_hyb(cx, cy, handle);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
     // A.spmv_cusparse_mp(cx, cy, handle);
     // A.spmv_cusparse(cx, cy, handle);
        A.spmv_cusparse_hyb(cx, cy, handle);
    timer.stop();
    if (cudaMemcpy(y, cy, props.m * sizeof(real_type), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("Error running cudaMemcpy function!");
    PRINT_RESULT("Measured MFLOP/s cuSPARSE:");

    cusparseDestroy(handle);
#endif

    A.release();

    free(x);
    free(y);

#ifdef HAVE_CUDA
    if (x) cudaFree(x);
    if (y) cudaFree(y);
#endif
}
