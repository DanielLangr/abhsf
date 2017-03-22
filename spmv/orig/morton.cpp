#include <algorithm>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <abhsf/utils/colors.h>
#include <abhsf/utils/math.h>
#include <abhsf/utils/matrix_properties.h>
#include <abhsf/utils/matrix_market_reader.h>
#include <abhsf/utils/timer.h>

using timer_type = chrono_timer<>;

using element_t = std::tuple<uint64_t, double>;
using elements_t = std::vector<element_t>;

uint64_t morton(uint32_t a, uint32_t b)
{
    uint64_t c = 0;

 // c = ((uint64_t)a << 26) + (uint64_t)b; // for uk-2002 matrix
    for (int i = 0; i < 32; i++)
        c |= (((uint64_t)a & (1UL << i)) << i) | (((uint64_t)b & (1UL << i)) << (i + 1));

    return c;
}

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

        assert (row < (1UL << 32));
        assert (col < (1UL << 32));

        if (row > col)
            L = true;
        if (col > row)
            U = true;

        if (props.type == matrix_type_t::BINARY) {
            elements.emplace_back(morton(row, col), 1.0);
        }
        else if (props.type == matrix_type_t::REAL) {
            if (val_re == 0.0) {
                if (warned == false) {
                    std::cout << red << "Matrix file contains zero elements." << reset << std::endl;
                    warned = true;
                }
            }
            else 
                elements.emplace_back(morton(row, col), 1.0);
        }
        else
            throw std::runtime_error("Unsupported matrix type!");
    }

    if (props.symmetry != matrix_symmetry_t::UNSYMMETRIC) {
        if ((L == true) && (U == true))
            throw std::runtime_error("Elements from both L and U parts stored for not-unsymmetric matrix.");

        std::cout << "Matrix symmetric part stored: " << ((L == true) ? "LOWER" : "UPPER") << std::endl;
    }

    std::cout << green << "... [DONE]" << reset << std::endl;
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

    // sort elements w.r.t. Morton code
    std::sort(elements.begin(), elements.end());

    // check for duplicates
    bool duplicates = false;
    for (long k = 1; k < elements.size(); k++)
        if (std::get<0>(elements[k]) == std::get<0>(elements[k - 1])) {
            duplicates = true;
            break;
        }
    if (duplicates)
        std::cout << red << "Matrix contains duplicite elements" << reset << std::endl;

    std::map<long, long> map;

    std::vector<uintmax_t> distances(elements.size() - 1);
    for (uintmax_t k = 1; k < elements.size(); k++) {
        intmax_t distance = std::get<0>(elements[k]) - std::get<0>(elements[k - 1]);
        assert(distance >= 0);
        distances[k] = distance;

        if (distance == 0)
            map[-1]++;
        else
            map[ceil_log2(distance)]++;
    }
    std::sort(distances.begin(), distances.end());

    std::cout << "Maximal Morton distances:" << std::endl;

    for (int k = 0; k < 50; k++) {
        uintmax_t distance = *(distances.end() - k - 1);
        std::cout
            << green << std::right << std::setw(20) << distance << reset
            << "; required bits: " << cyan << std::right << std::setw(8) << ceil_log2(distance) << reset
            << std::endl;
    }

    std::cout << "Count of distances according to their exponent:" << std::endl;
    for (auto iter = map.cbegin(); iter != map.cend(); ++iter)
        std::cout 
            << "Required bits: " << yellow << std::right << std::setw(8) << iter->first << reset
            << ": " << cyan << std::right << std::setw(20) << iter->second << reset
            << std::endl;
}
