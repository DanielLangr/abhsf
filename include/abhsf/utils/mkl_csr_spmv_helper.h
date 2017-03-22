#ifndef UTILS_MKL_CSR_SPMV_HELPER_H
#define UTILS_MKL_CSR_SPMV_HELPER_H

#include <mkl.h>

#include "restrict.h"

template <typename REAL_T>
class mkl_csr_spmv_helper;

template<>
struct mkl_csr_spmv_helper<float> 
{
    static void spmv(const MKL_INT m, const MKL_INT n, const float* RESTRICT a,
            const MKL_INT* RESTRICT ia, const MKL_INT* RESTRICT ja,
            const float* RESTRICT x, float* RESTRICT y)
    {
        static const char transa = 'N';
        static const float alpha = 1.0;
        static const char matdescra[6] = { 'G', ' ', ' ', 'C', ' ', ' ' };

        mkl_scsrmv(&transa, &m, &n, &alpha, matdescra, a, ja, ia, ia + 1, x, &alpha, y);
    }
};

template<>
struct mkl_csr_spmv_helper<double> 
{
    static void spmv(const MKL_INT m, const MKL_INT n, const double* RESTRICT a,
            const MKL_INT* RESTRICT ia, const MKL_INT* RESTRICT ja,
            const double* RESTRICT x, double* RESTRICT y)
    {
        static const char transa = 'N';
        static const double alpha = 1.0;
        static const char matdescra[6] = { 'G', ' ', ' ', 'C', ' ', ' ' };

        mkl_dcsrmv(&transa, &m, &n, &alpha, matdescra, a, ja, ia, ia + 1, x, &alpha, y);
    }
};

#endif
