#include <algorithm>
#include <cstdint>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <utils/colors.h>
#include <utils/matrix_properties.h>
#include <utils/matrix_market_reader.h>
#include <utils/timer.h>

using element_t = std::pair<uint32_t, uint32_t>;
using elements_t = std::vector<element_t>;

using timer_type = chrono_timer<>;

void read_mtx(const std::string& filename, elements_t& elements, matrix_properties& props) 
{
    std::cout << red << "Matrix market reader log..." << reset << std::endl;

    matrix_market_reader<> reader(&std::cout);
    reader.open(filename);
    
    props = reader.props();

    elements.reserve(props.nnz);
    for (uintmax_t k = 0; k < props.nnz; k++) {
        uintmax_t row, col;
        reader.next_element(&row, &col);
        elements.emplace_back(row, col);
    }

    std::cout << red << "... [DONE]" << reset << std::endl;
}

inline uint64_t block_index(const element_t& element, const int k)
{
    uint64_t block_row = (uint64_t)element.first >> k;
    uint64_t block_col = (uint64_t)element.second >> k;
    return (block_row << 32) + block_col;
}

using map_t = std::map<uint64_t, uint64_t>;

void process_block_size(elements_t& elements, const int k, map_t& map)
{
    assert(elements.size() > 0);

    auto comp = [k](const element_t& lhs, const element_t& rhs) {
        return block_index(lhs, k) < block_index(rhs, k);
    };

    std::sort(elements.begin(), elements.end(), comp);

    auto bi_1 = block_index(elements.front(), k);
    uintmax_t block_nnz = 1;
    for (auto iter = elements.cbegin() + 1; iter != elements.cend(); ++iter) {
        auto bi_2 = block_index(*iter, k);
        if (bi_2 != bi_1) {
            map[block_nnz]++;
            block_nnz = 0;
            bi_1 = bi_2;
        }
        block_nnz++;
    }
    map[block_nnz]++;
}

uintmax_t ceil_div(uintmax_t divident, uintmax_t divisor)
{
    assert(divisor > 0);

    if (divident % divisor)
        return divident / divisor + 1;
    else 
        return divident / divisor;
} 

uintmax_t ceil_log2(uintmax_t n)
{
    assert(n > 0);

    uintmax_t k = 0;
    while ((1UL << k) < n)
        k++;
    
    assert((1UL << k) >= n);
    assert((1UL << (k - 1)) < n);

    return k;
}

int main(int argc, char* argv[])
{

    elements_t elements;
    matrix_properties props;

    timer_type timer(timer_type::start_now);
    read_mtx(argv[1], elements, props);
    timer.stop();
    std::cout << "Matrix reading time: " << yellow << timer.seconds() << reset << " [s]" << std::endl;
    std::cout << std::endl;

    uintmax_t mm_csr0_coo1_min, s_csr0_coo1_min;
    uintmax_t mm_csr0_csr1_min, s_csr0_csr1_min;
    uintmax_t mm_csr0_bitmap1_min, s_csr0_bitmap1_min;
    uintmax_t mm_csr0_dense1_single_min, s_csr0_dense1_single_min;
    uintmax_t mm_csr0_dense1_double_min, s_csr0_dense1_double_min;
    uintmax_t mm_abhsf_single_min, s_abhsf_single_min;
    uintmax_t mm_abhsf_double_min, s_abhsf_double_min;

    map_t map; // how many blocks for particular block nonzero elements count ("histogram")
    for (int k = 1; k <= 10; k++) {
        const uintmax_t s = 1UL << k;
        std::cout << "Testing block size " << green << s << reset << "..." << std::endl;

        map.clear();
        process_block_size(elements, k, map);

        uintmax_t M = ceil_div(props.m, s); // number of block rows
        uintmax_t N = ceil_div(props.n, s); // number of block columns

        auto bits_N = ceil_log2(N);
        auto bits_block_nnz = 1UL << (2 * k);

        uintmax_t NNZ = 0; // number of nonzero blocks

        uintmax_t mm_coo1 = 0;
        uintmax_t mm_csr1 = 0;
        uintmax_t mm_bitmap1 = 0;
        uintmax_t mm_dense1_single = 0;
        uintmax_t mm_dense1_double = 0;
        uintmax_t mm_abhsf_single = 0;
        uintmax_t mm_abhsf_double = 0;

        uintmax_t nnz = 0; // check
        for (auto iter = map.cbegin(); iter != map.cend(); ++iter) {
            auto mm_block_coo = iter->first * 2 * k + bits_block_nnz;
            auto mm_block_csr = (s + 1) * bits_block_nnz + iter->first * k;
            auto mm_block_bitmap = s * s;
            auto block_zeros = s * s - iter->first;
            auto mm_block_dense_single = block_zeros * 32;
            auto mm_block_dense_double = block_zeros * 64;
            if (props.type == matrix_type_t::COMPLEX) {
                mm_block_dense_single *= 2;
                mm_block_dense_double *= 2;
            }
            auto mm_block_scheme = 2;

            mm_coo1 += iter->second * mm_block_coo;
            mm_csr1 += iter->second * mm_block_csr;
            mm_bitmap1 += iter->second * mm_block_bitmap;
            mm_dense1_single += iter->second * mm_block_dense_single;
            mm_dense1_double += iter->second * mm_block_dense_double;

            auto mm_block_min_single = std::min(std::min(mm_block_coo, mm_block_csr), mm_block_bitmap);
            auto mm_block_min_double = std::min(std::min(mm_block_coo, mm_block_csr), mm_block_bitmap);
            if (props.type != matrix_type_t::BINARY) {
                mm_block_min_single = std::min(mm_block_min_single, mm_block_dense_single);
                mm_block_min_double = std::min(mm_block_min_double, mm_block_dense_double);
            }
            mm_abhsf_single += iter->second * (mm_block_scheme + mm_block_min_single);
            mm_abhsf_double += iter->second * (mm_block_scheme + mm_block_min_double);

            NNZ += iter->second;
            nnz += iter->first * iter->second;
        }

        auto mm_csr0 = (M + NNZ) * bits_N;

        auto mm_csr0_coo1 = mm_csr0 + mm_coo1;
        auto mm_csr0_csr1 = mm_csr0 + mm_csr1;
        auto mm_csr0_bitmap1 = mm_csr0 + mm_bitmap1;
        auto mm_csr0_dense1_single = mm_csr0 + mm_dense1_single;
        auto mm_csr0_dense1_double = mm_csr0 + mm_dense1_double;
        mm_abhsf_single += mm_csr0;
        mm_abhsf_double += mm_csr0;

        std::cout << "Memory footprint in CSR-COO:       " << yellow
            << std::right << std::setw(20) << mm_csr0_coo1 << reset << " [bits]" << std::endl;
        std::cout << "Memory footprint in CSR-CSR:       " << yellow
            << std::right << std::setw(20) << mm_csr0_csr1 << reset << " [bits]" << std::endl;
        std::cout << "Memory footprint in CSR-bitmap:    " << yellow
            << std::right << std::setw(20) << mm_csr0_bitmap1 << reset << " [bits]" << std::endl;
        if (props.type != matrix_type_t::BINARY) {
            std::cout << "Memory footprint in CSR-dense(32): " << yellow
                << std::right << std::setw(20) << mm_csr0_dense1_single << reset << " [bits]" << std::endl;
            std::cout << "Memory footprint in CSR-dense(64): " << yellow
                << std::right << std::setw(20) << mm_csr0_dense1_double << reset << " [bits]" << std::endl;
        }
        std::cout << "Memory footprint in ABHSF(32):     " << yellow
            << std::right << std::setw(20) << mm_abhsf_single << reset << " [bits]" << std::endl;
        std::cout << "Memory footprint in ABHSF(64):     " << yellow
            << std::right << std::setw(20) << mm_abhsf_double << reset << " [bits]" << std::endl;

        if ((k == 1) || (mm_csr0_coo1 < mm_csr0_coo1_min)) {
            mm_csr0_coo1_min = mm_csr0_coo1;
            s_csr0_coo1_min = s;
        }
        if ((k == 1) || (mm_csr0_csr1 < mm_csr0_csr1_min)) {
            mm_csr0_csr1_min = mm_csr0_csr1;
            s_csr0_csr1_min = s;
        }
        if ((k == 1) || (mm_csr0_bitmap1 < mm_csr0_bitmap1_min)) {
            mm_csr0_bitmap1_min = mm_csr0_bitmap1;
            s_csr0_bitmap1_min = s;
        }
        if ((k == 1) || (mm_csr0_dense1_single < mm_csr0_dense1_single_min)) {
            mm_csr0_dense1_single_min = mm_csr0_dense1_single;
            s_csr0_dense1_single_min = s;
        }
        if ((k == 1) || (mm_csr0_dense1_double < mm_csr0_dense1_double_min)) {
            mm_csr0_dense1_double_min = mm_csr0_dense1_double;
            s_csr0_dense1_double_min = s;
        }
        if ((k == 1) || (mm_abhsf_single < mm_abhsf_single_min)) {
            mm_abhsf_single_min = mm_abhsf_single;
            s_abhsf_single_min = s;
        }
        if ((k == 1) || (mm_abhsf_double < mm_abhsf_double_min)) {
            mm_abhsf_double_min = mm_abhsf_double;
            s_abhsf_double_min = s;
        }

     // assert(nnz == props.nnz);
        if (nnz != props.nnz)
            throw std::runtime_error("Nonzero elements counts do not match!");
    }

    std::cout << red << "Minimal memory footprints: " << reset << std::endl;
    std::cout << "CSR-COO:       " << magenta
        << std::right << std::setw(20) << mm_csr0_coo1_min << reset << " [bits]" << reset
        << " for block size: " << magenta << std::setw(6) << s_csr0_coo1_min << reset << std::endl;
    std::cout << "CSR-CSR:       " << magenta
        << std::right << std::setw(20) << mm_csr0_csr1_min << reset << " [bits]" << reset
        << " for block size: " << magenta << std::setw(6) << s_csr0_csr1_min << reset << std::endl;
    std::cout << "CSR-bitmap:    " << magenta
        << std::right << std::setw(20) << mm_csr0_bitmap1_min << reset << " [bits]" << reset
        << " for block size: " << magenta << std::setw(6) << s_csr0_bitmap1_min << reset << std::endl;
    if (props.type != matrix_type_t::BINARY) {
        std::cout << "CSR-dense(32): " << magenta
            << std::right << std::setw(20) << mm_csr0_dense1_single_min << reset << " [bits]" << reset
            << " for block size: " << magenta << std::setw(6) << s_csr0_dense1_single_min << reset << std::endl;
        std::cout << "CSR-dense(64): " << magenta
            << std::right << std::setw(20) << mm_csr0_dense1_double_min << reset << " [bits]" << reset
            << " for block size: " << magenta << std::setw(6) << s_csr0_dense1_double_min << reset << std::endl;
    }
    std::cout << "ABHSF(32):     " << magenta
        << std::right << std::setw(20) << mm_abhsf_single_min << reset << " [bits]" << reset
        << " for block size: " << magenta << std::setw(6) << s_abhsf_single_min << reset << std::endl;
    std::cout << "ABHSF(64):     " << magenta
        << std::right << std::setw(20) << mm_abhsf_double_min << reset << " [bits]" << reset
        << " for block size: " << magenta << std::setw(6) << s_abhsf_double_min << reset << std::endl;

    uintmax_t mm_csr32 = (props.m + 1) * 32 + props.nnz * 32;
    std::cout << "CSR-32:        " << magenta
        << std::right << std::setw(20) << mm_csr32 << reset << " [bits]" << reset << std::endl;
}
