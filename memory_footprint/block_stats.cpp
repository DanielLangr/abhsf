#include <algorithm>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
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

int main(int argc, char* argv[])
{

    elements_t elements;
    matrix_properties props;

    timer_type timer(timer_type::start_now);
    read_mtx(argv[1], elements, props);
    timer.stop();
    std::cout << "Matrix reading time: " << yellow << timer.seconds() << reset << " [s]" << std::endl;
    std::cout << std::endl;

    const auto matname = matrix_name(argv[1]);
    std::ofstream f(matname + ".props");
    f << props.m << " " << props.n << " " << props.nnz << " "
        << static_cast<int>(props.type) << " " << static_cast<int>(props.symmetry) << std::endl;
    f.close();

    map_t map; // how many blocks for particular block nonzero elements count ("histogram")
    for (int k = 1; k <= 10; k++) {
        const uintmax_t s = 1UL << k;
        std::cout << "Testing block size: "
            << green << std::right << std::setw(6) << s << reset << std::endl;

        map.clear();
        process_block_size(elements, k, map);

        std::stringstream filename;
        filename << matname << "-" << s << ".bstats";
        std::ofstream f(filename.str());

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
