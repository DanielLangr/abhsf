#include <algorithm>
#include <cstdint>
#include <cassert>
//#include <cerrno>
#include <fstream>
#include <iostream>
#include <map>
//#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

//#include <sys/stat.h>

#include <abhsf/utils/colors.h>
#include <abhsf/utils/matrix_properties.h>
#include <abhsf/utils/matrix_market_reader.h>

using element_t = std::pair<uint32_t, uint32_t>;
using elements_t = std::vector<element_t>;

/*
std::vector<std::pair<uintmax_t, uintmax_t>> tested_bs_powers = {
    {1, 1}, {1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 2}, {3, 3},
    {3, 4}, {4, 3}, {4, 4}, {4, 5}, {5, 4}, {5, 5}, {6, 6}, {7, 7}
};
*/

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
            elements.emplace_back(row, col);
    }

    if ((props.symmetry != matrix_symmetry_t::UNSYMMETRIC) && (L == true) && (U == true))
        throw std::runtime_error("Elements from both L and U parts stored for not-unsymmetric matrix.");

    std::cout << red << "... [DONE]" << reset << std::endl;
}

inline uint64_t block_index_powers(const element_t& element, const int k, const int l)
{
    uint64_t block_row = (uint64_t)element.first >> k;
    uint64_t block_col = (uint64_t)element.second >> l;
    return (block_row << 32) + block_col;
}

using map_t = std::map<uint64_t, uint64_t>;

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
/*
std::string matrix_name(const std::string& path)
{
    assert (path.size() > 0);

    auto pos = path.find_last_of('/');
    if (pos == std::string::npos)
        pos = 0;
    const auto filename = path.substr(pos + 1);

    pos = filename.find_last_of('.');
    return filename.substr(0, pos);
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
        if ((props.symmetry != matrix_symmetry_t::UNSYMMETRIC) && (elem.first != elem.second))
            nnz_all++;
    }

    std::cout << "Nonzeros (.mtx):   " << magenta << std::right << std::setw(20) << props.nnz  << reset << std::endl; 
    std::cout << "Nonzeros (stored): " << magenta << std::right << std::setw(20) << nnz_stored << reset << std::endl; 
    std::cout << "Nonzeros (all):    " << magenta << std::right << std::setw(20) << nnz_all    << reset << std::endl; 

    // props file
    std::ofstream f("props");
    f << props.m << " " << props.n << " " << props.nnz << " "
        << nnz_stored << " " << nnz_all << " "
        << static_cast<int>(props.type) << " " << static_cast<int>(props.symmetry) << std::endl;
    f.close();

    map_t map; // how many blocks for particular block nonzero elements count ("histogram")
 // for (const auto bs_powers : tested_bs_powers) {
 //     const int k = bs_powers.first;
 //     const int l = bs_powers.second;
    for (int k = 1; k <= 8; k++) {
        for (int l = 1; l <= 8; l++) {
            const uintmax_t r = 1UL << k;
            const uintmax_t s = 1UL << l;

         // std::cout << "Testing block size: "
         //     << green << std::right << std::setw(6) << r << " x " << s << reset << std::endl;

            map.clear();
            process_block_size_powers(elements, k, l, map);

            // statistics file
            std::ofstream f(std::to_string(r) + "x" + std::to_string(s) + ".bstats");

            uintmax_t nnz = 0; // check
            for (auto iter = map.cbegin(); iter != map.cend(); ++iter) {
                f << iter->first << " " << iter->second << std::endl;
                nnz += iter->first * iter->second;
            }
            
            f.close();

         // assert(nnz == props.nnz);
            if (nnz != nnz_stored)
                throw std::runtime_error("Nonzero elements counts do not match!");
        }
    }
    map.clear();

    // rows/cols nnz (stored, all):
    std::vector<uintmax_t> rows_nnz_stored(props.m, 0);
    std::vector<uintmax_t> rows_nnz_all   (props.m, 0);
    std::vector<uintmax_t> cols_nnz_stored(props.n, 0);
    std::vector<uintmax_t> cols_nnz_all   (props.n, 0);

    for (auto& elem : elements) {
        rows_nnz_stored[elem.first ]++;
        cols_nnz_stored[elem.second]++;

        rows_nnz_all[elem.first ]++;
        cols_nnz_all[elem.second]++;
        if ((props.symmetry != matrix_symmetry_t::UNSYMMETRIC) && (elem.first != elem.second)) {
            rows_nnz_all[elem.second]++;
            cols_nnz_all[elem.first ]++;
        }
    }
    
    // check
    if (std::accumulate(rows_nnz_stored.cbegin(), rows_nnz_stored.cend(), uint64_t(0)) != nnz_stored)
        throw std::runtime_error("Stored nonzero elements coutns do not match!");
    if (std::accumulate(cols_nnz_stored.cbegin(), cols_nnz_stored.cend(), uint64_t(0)) != nnz_stored)
        throw std::runtime_error("Stored nonzero elements coutns do not match!");
    if (std::accumulate(rows_nnz_all.cbegin(), rows_nnz_all.cend(), uint64_t(0)) != nnz_all)
        throw std::runtime_error("All nonzero elements coutns do not match!");
    if (std::accumulate(cols_nnz_all.cbegin(), cols_nnz_all.cend(), uint64_t(0)) != nnz_all)
        throw std::runtime_error("All nonzero elements coutns do not match!");


    // store
    std::ofstream frs("rows_nnz_stored");
    for (auto row_nnz : rows_nnz_stored)
        frs << row_nnz << std::endl;
    frs.close();

    std::ofstream fcs("cols_nnz_stored");
    for (auto col_nnz : cols_nnz_stored)
        fcs << col_nnz << std::endl;
    fcs.close();

    std::ofstream fra("rows_nnz_all");
    for (auto row_nnz : rows_nnz_all)
        fra << row_nnz << std::endl;
    fra.close();

    std::ofstream fca("cols_nnz_all");
    for (auto col_nnz : cols_nnz_all)
        fca << col_nnz << std::endl;
    fca.close();
}
