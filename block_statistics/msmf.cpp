#include <iomanip>
#include <iostream>
#include <limits>

#include <abhsf/msmf.h>
#include <abhsf/utils/colors.h>
#include <abhsf/utils/math.h>
#include <abhsf/utils/matrix_properties.h>

#include "stats.h"

void print_single(const std::string& format, uintmax_t bits) 
{
    std::cout
        << "Memory footprint in " << std::left << std::setw(16) << (format + ": ") 
        << yellow << std::right << std::setw(20) << bits << reset << " [bits]" << std::endl;
}

int main()
{
    matrix_block_stats stats;
    stats.read();

    std::cout << "Checking block statistics... ";
    if (stats.check())
        std::cout << green << "[OK]";
    else 
        std::cout << red << "[FAILED]";
    std::cout << reset << std::endl;

    const auto& props = stats.props();
    const bool is_binary = (props.type == matrix_type_t::BINARY);

    uintmax_t bits_per_single_element;
    switch (props.type) {
        case matrix_type_t::INTEGER:
        case matrix_type_t::REAL:
            bits_per_single_element = 32;
            break;

        case matrix_type_t::COMPLEX:
            bits_per_single_element = 64;
            break;

        default:
            bits_per_single_element = 0;
    }

    std::ofstream f("msmf");

    uintmax_t msmf_coo32 = props.nnz * 32 * 2;
    uintmax_t msmf_csr32 = (props.m + 1 + props.nnz) * 32;

    for (const auto& temp : stats.stats()) {
        const uintmax_t r = temp.first.first;
        const uintmax_t s = temp.first.second;
        std::cout << "Processed block size: " << cyan << r << " x " << s << reset << "..." << std::endl;

        uintmax_t M = ceil_div(props.m, r); // number of block rows
        uintmax_t N = ceil_div(props.n, s); // number of block columns

        uintmax_t NNZ = 0; // number of nonzero blocks
        for (const auto& bstats : temp.second) 
            NNZ += bstats.second;

        uintmax_t msmf_csr0 = M * ceil_log2(N + 1) + NNZ * ceil_log2(N);

        uintmax_t msmf_coo = msmf_csr0;
        uintmax_t msmf_csr = msmf_csr0;
        uintmax_t msmf_bitmap = msmf_csr0;
        uintmax_t msmf_dense_single = msmf_csr0, msmf_dense_double = msmf_csr0;
        uintmax_t msmf_abhsf_single = msmf_csr0, msmf_abhsf_double = msmf_csr0;

        if (is_binary) {
            msmf_dense_single = std::numeric_limits<uintmax_t>::max();
            msmf_dense_double = std::numeric_limits<uintmax_t>::max();
        }

        for (const auto& bstats : temp.second) {
            uintmax_t block_nnz = bstats.first;
            uintmax_t n_blocks = bstats.second;

            msmf_coo += block_coo_msmf(r, s, block_nnz) * n_blocks;
            msmf_csr += block_csr_msmf(r, s, block_nnz) * n_blocks;
            msmf_bitmap += block_bitmap_msmf(r, s) * n_blocks;
            if (!is_binary) {
                msmf_dense_single += block_dense_msmf(r, s, block_nnz, bits_per_single_element) * n_blocks;
                msmf_dense_double += block_dense_msmf(r, s, block_nnz, bits_per_single_element * 2) * n_blocks;
            }
            msmf_abhsf_single
                += (block_min_msmf(r, s, block_nnz, bits_per_single_element, is_binary) + 2) * n_blocks;
            msmf_abhsf_double
                += (block_min_msmf(r, s, block_nnz, bits_per_single_element * 2, is_binary) + 2) * n_blocks;
        }

        print_single("CSR-COO", msmf_coo);
        print_single("CSR-CSR", msmf_csr);
        print_single("CSR-bitmap", msmf_bitmap);
        if (!is_binary) print_single("CSR-dense(32)", msmf_dense_single);
        if (!is_binary) print_single("CSR-dense(64)", msmf_dense_double);
        print_single("CSR-ABHSF(32)", msmf_abhsf_single);
        print_single("CSR-ABHSF(64)", msmf_abhsf_double);

        f << r << " " << s << " "
            << msmf_coo32 << " " << msmf_csr32 << " " 
            << msmf_coo << " " << msmf_csr << " " << msmf_bitmap << " "
            << msmf_dense_single << " " << msmf_dense_double << " "
            << msmf_abhsf_single << " " << msmf_abhsf_double
            << std::endl;
    }

    f.close();
}
