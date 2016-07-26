#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mmf.h"

#include <abhsf/utils/colors.h>

void process_precison(mmf_t::precision_t precision)
{
    mmf_t mmf;
    mmf.load(precision);
    auto const& map = mmf.map();

    std::string filename;
    if (precision == mmf_t::precision_t::SINGLE)
        filename = "mmf-bss-single";
    else
        filename = "mmf-bss-double";
    std::ofstream f(filename);

    // header
    f << "h w ";
    for (auto const& matrix : map)
        f << matrix.first << " ";
    f << std::endl;

    // minima for block sizes
    for (uintmax_t h = 2; h <= 256; h *= 2) {
        for (uintmax_t w = 2; w <= 256; w *= 2) {
            std::cout << "Processing block size " << cyan << h << "x" << w << reset << std::endl;
            f << h << " " << w << " ";
            mmf_t::block_size_t block_size = std::make_pair(h, w);

            for (auto const& matrix : map) {
                auto const& matrix_entry = matrix.second;

                auto iter = matrix_entry.find(block_size);
                if (iter == matrix_entry.end())
                    throw std::runtime_error("Invalid block size!");

                uintmax_t min = std::numeric_limits<uintmax_t>::max();
                for (int k = 2; k <= 6; k++)
                    min = std::min(min, (iter->second)[k]);
                f << min << " ";
            }

            f << std::endl;
        }
    }

    f.close();
}

int main()
{
    process_precison(mmf_t::precision_t::SINGLE);
    process_precison(mmf_t::precision_t::DOUBLE);
}
