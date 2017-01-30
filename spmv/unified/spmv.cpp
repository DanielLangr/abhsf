#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

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
            std::sort(elements.begin(), elements.end());

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
            std::sort(elements.begin(), elements.end());

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
/*
        void spmv(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
#pragma omp parallel for
            for (index_type row = 0; row < m_; row++) 
                for (index_type k = ia_[row]; k < ia_[row + 1]; k++)
                    y[row] += a_[k] * x[ja_[k]];
        }
*/

#ifdef HAVE_MKL
        void spmv_mkl(const real_type* RESTRICT x, real_type* RESTRICT y)
        {
            static const char transa = 'N';
            static const double alpha = 1.0;
            static const char matdescra[6] = { 'G', ' ', ' ', 'C', ' ', ' ' };
            mkl_dcoomv(&transa, &m_, &n_, &alpha, matdescra, 
                    a_.data(), ia_.data(), ja_.data(), &nnz, x, &alpha, y);
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
        props.nnz = props.m;

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
    posix_memalign((void**)(&y), 64, props.m * sizeof(real_type));
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

#ifdef HAVE_CUDA
    cusparseHandle_t handle;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("Error running cusparseCreate function!");

    csr.to_hyb(handle);

    std::fill(y, y + props.m, 0.0);
    if (cudaMemcpy(cy, y, props.m * sizeof(real_type), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Error running cudaMemcpy function!");
    for (int iter = 0; iter < warmup_iters; iter++) 
     // csr.spmv_cusparse_mp(cx, cy, handle); // available since CUDA 8.0
     // csr.spmv_cusparse(cx, cy, handle);
        csr.spmv_cusparse_hyb(cx, cy, handle);
    timer.start();
    for (int iter = 0; iter < n_iters; iter++) 
     // csr.spmv_cusparse_mp(cx, cy, handle);
     // csr.spmv_cusparse(cx, cy, handle);
        csr.spmv_cusparse_hyb(cx, cy, handle);
    timer.stop();
    if (cudaMemcpy(y, cy, props.m * sizeof(real_type), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("Error running cudaMemcpy function!");
    PRINT_RESULT("Measured MFLOP/s cuSPARSE CSR:");

    cusparseDestroy(handle);
#endif

    csr.release();

    free(x);
    free(y);

#ifdef HAVE_CUDA
    if (x) cudaFree(x);
    if (y) cudaFree(y);
#endif
}
