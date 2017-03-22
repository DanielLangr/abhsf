#ifndef UTILS_COO2CSR_H
#define UTILS_COO2CSR_H

#include <cassert>

#include <parallel/algorithm>

#include "matrix_properties.h"

template <typename ELEMENTS_T, typename A, typename IA, typename JA>
void coo2csr(ELEMENTS_T& elements, const long m, A& a, IA& ia, JA& ja)
{
    __gnu_parallel::sort(elements.begin(), elements.end());

    long nnz = elements.size();
    a.resize(nnz);
    ia.resize(m + 1);
    ja.resize(nnz);

    for (long k = 0; k < nnz; k++) {
        a[k] = std::get<2>(elements[k]);
        ja[k] = std::get<1>(elements[k]);
    }

    ia[0] = 0;
    long k = 0;
    long row = 0;

    while (k < nnz) {
        while ((k < nnz) && (row == std::get<0>(elements[k])))
            k++;

        row++;
        ia[row] = k;
    }

    assert(ia[m] == nnz);
}

#endif
