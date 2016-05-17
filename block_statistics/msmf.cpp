#include <iomanip>
#include <iostream>
#include <limits>

#include <abhsf/msmf.h>
#include <abhsf/utils/colors.h>
#include <abhsf/utils/math.h>
#include <abhsf/utils/matrix_properties.h>

#include "stats.h"

void treat_min(uintmax_t tested, uintmax_t& min, uintmax_t s, uintmax_t& min_s)
{
    if (tested < min) {
        min = tested;
        min_s = s;
    }
}

void print_single(const std::string& format, uintmax_t bits) 
{
    std::cout
        << "Memory footprint in " << std::left << std::setw(16) << (format + ": ") 
        << yellow << std::right << std::setw(20) << bits << reset << " [bits]" << std::endl;
}

void print_min(const std::string& format, uintmax_t bits, uintmax_t s)
{
    std::cout
        << "Memory footprint in " << std::left << std::setw(16) << (format + ": ") 
        << cyan << std::right << std::setw(20) << bits << reset << " [bits], "
        << "block size: " << magenta << cyan << std::right << std::setw(6) << s << reset << std::endl;
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

    uintmax_t min_msmf_coo, min_msmf_csr, min_msmf_bitmap, min_msmf_dense_single,
              min_msmf_dense_double, min_msmf_abhsf_single, min_msmf_abhsf_double;
    min_msmf_coo = min_msmf_csr = min_msmf_bitmap = min_msmf_dense_single = min_msmf_dense_double
        = min_msmf_abhsf_single = min_msmf_abhsf_double = std::numeric_limits<uintmax_t>::max();
    uintmax_t min_coo_s, min_csr_s, min_bitmap_s, min_dense_single_s,
              min_dense_double_s, min_abhsf_single_s, min_abhsf_double_s;

    for (const auto& temp : stats.stats()) {
        const uintmax_t s = temp.first;
        std::cout << "Processed block size: " << cyan << s << reset << "..." << std::endl;

        uintmax_t M = ceil_div(props.m, s); // number of block rows
        uintmax_t N = ceil_div(props.n, s); // number of block columns

        uintmax_t NNZ = 0; // number of nonzero blocks
        for (const auto& bstats : temp.second) 
            NNZ += bstats.second;

        uintmax_t mmfs_csr0 = ceil_log2(N) * (M + NNZ); 

        uintmax_t msmf_coo = mmfs_csr0;
        uintmax_t msmf_csr = mmfs_csr0;
        uintmax_t msmf_bitmap = mmfs_csr0;
        uintmax_t msmf_dense_single = mmfs_csr0, msmf_dense_double = mmfs_csr0;
        uintmax_t msmf_abhsf_single = mmfs_csr0, msmf_abhsf_double = mmfs_csr0;

        for (const auto& bstats : temp.second) {
            uintmax_t block_nnz = bstats.first;
            uintmax_t n_blocks = bstats.second;

            msmf_coo += block_coo_msmf(s, s, block_nnz) * n_blocks;
            msmf_csr += block_csr_msmf(s, s, block_nnz) * n_blocks;
            msmf_bitmap += block_bitmap_msmf(s, s) * n_blocks;
            if (!is_binary) {
                msmf_dense_single += block_dense_msmf(s, s, block_nnz, bits_per_single_element) * n_blocks;
                msmf_dense_double += block_dense_msmf(s, s, block_nnz, bits_per_single_element * 2) * n_blocks;
            }
            msmf_abhsf_single
                += (block_min_msmf(s, s, block_nnz, bits_per_single_element, is_binary) + 2) * n_blocks;
            msmf_abhsf_double
                += (block_min_msmf(s, s, block_nnz, bits_per_single_element * 2, is_binary) + 2) * n_blocks;
        }

        // treat minima
        treat_min(msmf_coo, min_msmf_coo, s, min_coo_s);
        treat_min(msmf_csr, min_msmf_csr , s, min_csr_s);
        treat_min(msmf_bitmap, min_msmf_bitmap, s, min_bitmap_s);
        if (!is_binary) {
            treat_min(msmf_dense_single, min_msmf_dense_single, s, min_dense_single_s);
            treat_min(msmf_dense_double, min_msmf_dense_double, s, min_dense_double_s);
        }
        treat_min(msmf_abhsf_single, min_msmf_abhsf_single, s, min_abhsf_single_s);
        treat_min(msmf_abhsf_double, min_msmf_abhsf_double, s, min_abhsf_double_s);

        print_single("CSR-COO", msmf_coo);
        print_single("CSR-CSR", msmf_csr);
        print_single("CSR-bitmap", msmf_bitmap);
        if (!is_binary) print_single("CSR-dense(32)", msmf_dense_single);
        if (!is_binary) print_single("CSR-dense(64)", msmf_dense_double);
        print_single("CSR-ABHSF(32)", msmf_abhsf_single);
        print_single("CSR-ABHSF(64)", msmf_abhsf_double);
    }

    uintmax_t msmf_csr32 = (props.m + 1 + props.nnz) * 32;

    // print minima
    std::cout << magenta << "MSMF MINIMA:" << reset << std::endl;
    print_min("CSR-COO", min_msmf_coo, min_coo_s);
    print_min("CSR-CSR", min_msmf_csr, min_csr_s);
    print_min("CSR-bitmap", min_msmf_bitmap, min_bitmap_s);
    if (!is_binary) print_min("CSR-dense(32)", min_msmf_dense_single, min_dense_single_s);
    if (!is_binary) print_min("CSR-dense(64)", min_msmf_dense_double, min_dense_double_s);
    print_min("CSR-ABHSF(32)", min_msmf_abhsf_single, min_abhsf_single_s);
    print_min("CSR-ABHSF(64)", min_msmf_abhsf_double, min_abhsf_double_s);
    print_min("CSR-32", msmf_csr32, 0);

    std::ofstream f_msmf("msmf");
    f_msmf
        << min_msmf_coo << " " << min_coo_s << std::endl
        << min_msmf_csr << " " << min_csr_s << std::endl
        << min_msmf_bitmap << " " << min_bitmap_s << std::endl
        << min_msmf_dense_single << " " << min_dense_single_s << std::endl
        << min_msmf_dense_double << " " << min_dense_double_s << std::endl
        << min_msmf_abhsf_single << " " << min_abhsf_single_s << std::endl
        << min_msmf_abhsf_double << " " << min_abhsf_double_s << std::endl
        << msmf_csr32 << std::endl;
    f_msmf.close();

    std::cout << magenta << "MMF MINIMA SINGLE:" << reset << std::endl;
    uintmax_t vmf = props.nnz * bits_per_single_element;
    print_min("CSR-COO", min_msmf_coo + vmf, min_coo_s);
    print_min("CSR-CSR", min_msmf_csr + vmf, min_csr_s);
    print_min("CSR-bitmap", min_msmf_bitmap + vmf, min_bitmap_s);
    if (!is_binary) print_min("CSR-dense(32)", min_msmf_dense_single + vmf, min_dense_single_s);
    if (!is_binary) print_min("CSR-dense(64)", min_msmf_dense_double + vmf, min_dense_double_s);
    print_min("CSR-ABHSF(32)", min_msmf_abhsf_single + vmf, min_abhsf_single_s);
    print_min("CSR-ABHSF(64)", min_msmf_abhsf_double + vmf, min_abhsf_double_s);
    print_min("CSR-32", msmf_csr32 + vmf, 0);

    std::ofstream f_mmf32("mmf32");
    f_mmf32
        << min_msmf_coo + vmf << " " << min_coo_s << std::endl
        << min_msmf_csr + vmf << " " << min_csr_s << std::endl
        << min_msmf_bitmap + vmf << " " << min_bitmap_s << std::endl
        << min_msmf_dense_single + vmf << " " << min_dense_single_s << std::endl
        << min_msmf_dense_double + vmf << " " << min_dense_double_s << std::endl
        << min_msmf_abhsf_single + vmf << " " << min_abhsf_single_s << std::endl
        << min_msmf_abhsf_double + vmf << " " << min_abhsf_double_s << std::endl
        << msmf_csr32 + vmf << std::endl;
    f_mmf32.close();

    std::cout << magenta << "MMF MINIMA DOUBLE:" << reset << std::endl;
    vmf = props.nnz * bits_per_single_element * 2;
    print_min("CSR-COO", min_msmf_coo + vmf, min_coo_s);
    print_min("CSR-CSR", min_msmf_csr + vmf, min_csr_s);
    print_min("CSR-bitmap", min_msmf_bitmap + vmf, min_bitmap_s);
    if (!is_binary) print_min("CSR-dense(32)", min_msmf_dense_single + vmf, min_dense_single_s);
    if (!is_binary) print_min("CSR-dense(64)", min_msmf_dense_double + vmf, min_dense_double_s);
    print_min("CSR-ABHSF(32)", min_msmf_abhsf_single + vmf, min_abhsf_single_s);
    print_min("CSR-ABHSF(64)", min_msmf_abhsf_double + vmf, min_abhsf_double_s);
    print_min("CSR-32", msmf_csr32 + vmf, 0);

    std::ofstream f_mmf64("mmf64");
    f_mmf64
        << min_msmf_coo + vmf << " " << min_coo_s << std::endl
        << min_msmf_csr + vmf << " " << min_csr_s << std::endl
        << min_msmf_bitmap + vmf << " " << min_bitmap_s << std::endl
        << min_msmf_dense_single + vmf << " " << min_dense_single_s << std::endl
        << min_msmf_dense_double + vmf << " " << min_dense_double_s << std::endl
        << min_msmf_abhsf_single + vmf << " " << min_abhsf_single_s << std::endl
        << min_msmf_abhsf_double + vmf << " " << min_abhsf_double_s << std::endl
        << msmf_csr32 + vmf << std::endl;
    f_mmf64.close();
}
