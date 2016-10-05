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

/*
inline uint64_t block_index_powers(const element_t& element, const int k, const int l)
{
    uint64_t block_row = (uint64_t)element.first >> k;
    uint64_t block_col = (uint64_t)element.second >> l;
    return (block_row << 32) + block_col;
}

void process_block_size_powers(elements_t& elements, const int k, const int l, map_t& map)
{
    assert(elements.size() > 0);

    auto comp = [k, l](const element_t& lhs, const element_t& rhs) {
        return block_index_powers(lhs, k, l) < block_index_powers(rhs, k, l);
    };

    std::sort(elements.begin(), elements.end(), comp);

    auto bi_1 = block_index_powers(elements.front(), k, l);
    uintmax_t block_nnz = 1;
    for (auto iter = elements.cbegin() + 1; iter != elements.cend(); ++iter) {
        auto bi_2 = block_index_powers(*iter, k, l);
        if (bi_2 != bi_1) {
            map[block_nnz]++;
            block_nnz = 0;
            bi_1 = bi_2;
        }
        block_nnz++;
    }
    map[block_nnz]++;
}
*/

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
}
