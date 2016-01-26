#include <omp.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include <matrix_market_reader.h>
#include <timer.h>

using timer_type = chrono_timer<>;

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

    switch (reader.type) {
        case matrix_market_reader::INTEGER:
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
        double re, im;
        reader.next_element(&row, &col, &re, &im);

        rows[k] = row;
        cols[k] = col;

        switch (reader.type) {
            case matrix_market_reader::INTEGER:
            case matrix_market_reader::REAL:
                vals_re[k] = re;
                break;

            case matrix_market_reader::COMPLEX:
                vals_re[k] = re;
                vals_im[k] = im;
                break;

            default: // prevents compiler warning
                break;
        }
    }

    timer.stop();
    std::cout << "Matrix reading time: " << timer.seconds() << " [s]" << std::endl;
}
