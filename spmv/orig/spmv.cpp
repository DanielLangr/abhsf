#include <algorithm>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

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

inline uint64_t block_row(const element_t& element, const int k)
{
    return ((uint64_t)(std::get<0>(element))) >> k;
}

inline uint64_t block_col(const element_t& element, const int l)
{
    return ((uint64_t)(std::get<1>(element))) >> l;
}

inline uint64_t inblock_row(const element_t& element, const int k)
{
    return ((uint64_t)(std::get<0>(element))) & ((1UL << k) - 1);
}

inline uint64_t inblock_col(const element_t& element, const int l)
{
    return ((uint64_t)(std::get<1>(element))) & ((1UL << l) - 1);
}

void sort_wrt_blocks_lex(elements_t& elements, const int k, const int l)
{
    auto comp = [k, l](const element_t& lhs, const element_t& rhs) {
        if (block_row(lhs, k) < block_row(rhs, k))
            return true;
        else if (block_row(lhs, k) > block_row(rhs, k))
            return false;
        // same block row

        if (block_col(lhs, l) < block_col(rhs, l))
            return true;
        else if (block_col(lhs, l) > block_col(rhs, l))
            return false;
        // same block

        if (inblock_row(lhs, k) < inblock_row(rhs, k))
            return true;
        else if (inblock_row(lhs, k) > inblock_row(rhs, k))
            return false;
        // same row within same block

        return inblock_col(lhs, l) < inblock_col(rhs, l);
    };

    std::sort(elements.begin(), elements.end(), comp);
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

    std::cout << "sizeof(MKL_INT): " << sizeof(MKL_INT) << std::endl;

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

    static const int n_iters = 200;

    timer_type timer(timer_type::start_now);

    for (int iter = 0; iter < n_iters; iter++) 
        mkl_cspblas_dcsrgemv(&transa, &m, a.data(), ia.data(), ja.data(), x.data(), y.data());

    timer.stop();

    uintmax_t n_flops = 2 * nnz_all * n_iters;
    double mega_flops = double(n_flops) / 1.0e6 / timer.seconds();
    std::cout << "Measured MFLOP/S Intel MKL: " << yellow << mega_flops << reset << std::endl;

    // naive CSR32
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
/*
    a.clear();
    a.shrink_to_fit();
    ia.clear();
    ia.shrink_to_fit();
    ja.clear();
    ja.shrink_to_fit();

    // min-fixed coo
    int k = 4; // Freescale1: 16x16
    int l = 4;
    sort_wrt_blocks_lex(elements, k, l);

    std::vector<double> coo_vals(nnz_stored);
    uint64_t coo_str_size = (uint64_t)(k + l) * nnz_stored / 8 + 1;
    std::vector<uint8_t> coo_str(coo_str_size);

    auto bi_2 = block_index_powers(elements.front(), k, l);
    for (auto iter = elements.cbegin() + 1; iter != elements.cend(); ++iter) {
        auto bi_2 = block_index_powers(*iter, k, l);
        if (bi_2 != bi_1) {
        }
    }
*/
}
