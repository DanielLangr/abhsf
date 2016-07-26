#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "mmf.h"

#include <abhsf/utils/colors.h>

void process_precison(mmf_t::precision_t precision)
{
    mmf_t mmf;
    mmf.load(precision);
    auto const& map = mmf.map();

    std::string filename;
    if (precision == mmf_t::precision_t::SINGLE)
        filename = "mmf-schemes-single";
    else
        filename = "mmf-schemes-double";
    std::ofstream f(filename);

    for (auto const& matrix : map) {
        std::cout << "Processing matrix " << cyan << matrix.first << reset << std::endl;

        f << matrix.first << " ";

        auto const& matrix_entry = matrix.second;
        
        // for all schemes
        for (int k = 0; k < 8; k++) {
            // find min across block sizes
            uintmax_t min = std::numeric_limits<uintmax_t>::max();
            for (auto const& bs_record : matrix_entry) 
                min = std::min(min, (bs_record.second)[k]);
            f << min << " ";
        }

        mmf_t::block_size_t opt_block_size;
        uintmax_t opt = mmf.block_opt(matrix.first, &opt_block_size);
        f << opt_block_size.first << " " << opt_block_size.second << std::endl;
    }

    f.close();
}

int main()
{
    process_precison(mmf_t::precision_t::SINGLE);
    process_precison(mmf_t::precision_t::DOUBLE);
}
