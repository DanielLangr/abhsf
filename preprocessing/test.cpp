#include <omp.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include <matrix_market_reader.h>
#include <timer.h>

#include <aqsort.h>

using timer_type = chrono_timer<>;

const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string reset("\033[0m");

int main(int argc, char* argv[])
{
    using matrix_market_reader = matrix_market_reader<>;

    timer_type timer(timer_type::start_now);

    std::cout << "Reading matrix file: " << argv[1] << std::endl;
    matrix_market_reader reader(&std::cout);
    reader.open(argv[1]);

    uintmax_t nnz = reader.nnz;
    std::vector<uint32_t> rows(nnz), cols(nnz);
    std::vector<double> vals_re, vals_im;
    std::vector<intmax_t> vals_int;

    switch (reader.type) {
        case matrix_market_reader::INTEGER:
            vals_int.resize(nnz);
            break;

        case matrix_market_reader::REAL:
            vals_re.resize(nnz);
            break;

        case matrix_market_reader::COMPLEX:
            vals_re.resize(nnz);
            vals_im.resize(nnz);
            break;

        default: // prevents compiler warning
            break;
    }

    for (uintmax_t k = 0; k < nnz; k++) {
        uintmax_t row, col;
        double val_re, val_im;
        intmax_t val_int;
        reader.next_element(&row, &col, &val_re, &val_im, &val_int);

        rows[k] = row;
        cols[k] = col;

        switch (reader.type) {
            case matrix_market_reader::INTEGER:
                vals_int[k] = val_int;
                break;

            case matrix_market_reader::REAL:
                vals_re[k] = val_re;
                break;

            case matrix_market_reader::COMPLEX:
                vals_re[k] = val_re;
                vals_im[k] = val_im;
                break;

            default: // prevents compiler warning
                break;
        }
    }

    timer.stop();
    std::cout << "Matrix reading time: " << yellow << timer.seconds() << reset << " [s]" << std::endl;

    const uintmax_t bsk = std::atoi(argv[2]);
    std::cout << "Block size power: " << cyan << bsk << reset << std::endl;
    std::cout << "Block size: " << cyan << (1 << bsk) << reset <<std::endl;

    int num_threads = std::atoi(argv[3]);
    std::cout << "Number of threads: " << cyan << num_threads << reset << std::endl;

    auto comp = [&rows, &cols, bsk] (std::size_t i, std::size_t j) {
        uintmax_t Ii = rows[i] >> bsk;
        uintmax_t Ij = rows[j] >> bsk;
        uintmax_t Ji = cols[i] >> bsk;
        uintmax_t Jj = cols[j] >> bsk;

        if (Ii < Ij) return true;
        if ((Ii == Ij) && (Ji < Jj)) return true;
        return false;
    };

    auto swap_pat = [&rows, &cols] (std::size_t i, std::size_t j) {
        std::swap(rows[i], rows[j]);
        std::swap(cols[i], cols[j]);
    };

    auto swap_int = [&rows, &cols, &vals_int] (std::size_t i, std::size_t j) {
        std::swap(rows[i], rows[j]);
        std::swap(cols[i], cols[j]);
        std::swap(vals_int[i], vals_int[j]);
    };

    auto swap_real = [&rows, &cols, &vals_re] (std::size_t i, std::size_t j) {
        std::swap(rows[i], rows[j]);
        std::swap(cols[i], cols[j]);
        std::swap(vals_re[i], vals_re[j]);
    };

    auto swap_comp = [&rows, &cols, &vals_re, &vals_im] (std::size_t i, std::size_t j) {
        std::swap(rows[i], rows[j]);
        std::swap(cols[i], cols[j]);
        std::swap(vals_re[i], vals_re[j]);
        std::swap(vals_im[i], vals_im[j]);
    };

    omp_set_num_threads(num_threads);

    timer.start();
    switch (reader.type) {
        case matrix_market_reader::PATTERN:
            #pragma omp parallel 
            aqsort::parallel_sort(nnz, &comp, &swap_pat);
            break;

        case matrix_market_reader::INTEGER:
            #pragma omp parallel 
            aqsort::parallel_sort(nnz, &comp, &swap_int);
            break;

        case matrix_market_reader::REAL:
            #pragma omp parallel 
            aqsort::parallel_sort(nnz, &comp, &swap_real);
            break;

        case matrix_market_reader::COMPLEX:
            #pragma omp parallel 
            aqsort::parallel_sort(nnz, &comp, &swap_comp);
            break;
    }
    timer.stop();
    std::cout << "Sorting of elements: " << yellow << timer.seconds() << reset << std::endl;
}
