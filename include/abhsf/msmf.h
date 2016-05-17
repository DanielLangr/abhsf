#ifndef ABHSF_MSMF_H
#define ABHSF_MSMF_H

#include <algorithm>
#include <cstdint>

#include "utils/math.h"

inline uintmax_t block_coo_msmf(uintmax_t m, uintmax_t n, uintmax_t nnz)
{
    return 
        ceil_log2(m * n)                     // need to store number of nonzeros 
      + nnz * (ceil_log2(m) + ceil_log2(n)); // row and column indexes
}

inline uintmax_t block_csr_msmf(uintmax_t m, uintmax_t n, uintmax_t nnz)
{
    return 
        m * ceil_log2(m * n) // row pointers
      + nnz * ceil_log2(n);  // column pointers
}

inline uintmax_t block_bitmap_msmf(uintmax_t m, uintmax_t n)
{
    return m * n;
}

inline uintmax_t block_dense_msmf(uintmax_t m, uintmax_t n, uintmax_t nnz, uintmax_t bits_per_element)
{
    assert (m * n >= nnz);

    return (m * n - nnz) * bits_per_element;
}

inline uintmax_t block_min_msmf(uintmax_t m, uintmax_t n, uintmax_t nnz, uintmax_t bits_per_element, bool is_binary)
{
    auto coo = block_coo_msmf(m, n, nnz);
    auto csr = block_csr_msmf(m, n, nnz);
    auto bitmap = block_bitmap_msmf(m, n);

    if (is_binary) { // do not consider dense scheme for binary matirces
        return std::min(std::min(coo, csr), bitmap);
    }
    else {
        auto dense = block_dense_msmf(m, n, nnz, bits_per_element);
        return std::min(std::min(coo, csr), std::min(bitmap, dense));
    }
}

#endif
