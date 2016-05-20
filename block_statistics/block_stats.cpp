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

std::vector<std::pair<uintmax_t, uintmax_t>> tested_bs_powers = {
    {1, 1}, {1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 2}, {3, 3},
    {3, 4}, {4, 3}, {4, 4}, {4, 5}, {5, 4}, {5, 5}, {6, 6}, {7, 7}
};

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

    // props file
    std::ofstream f("props");
    f << props.m << " " << props.n << " " << props.nnz << " "
        << static_cast<int>(props.type) << " " << static_cast<int>(props.symmetry) << std::endl;
    f.close();

    map_t map; // how many blocks for particular block nonzero elements count ("histogram")
    for (const auto bs_powers : tested_bs_powers) {
        const int k = bs_powers.first;
        const int l = bs_powers.second;
        const uintmax_t r = 1UL << k;
        const uintmax_t s = 1UL << l;

        std::cout << "Testing block size: "
            << green << std::right << std::setw(6) << r << " x " << s << reset << std::endl;

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
        if (nnz != props.nnz)
            throw std::runtime_error("Nonzero elements counts do not match!");
    }
}
