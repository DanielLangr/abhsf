#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <matrix_market_reader.h>
#include <timer.h>

using timer_type = chrono_timer<>;

const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string reset("\033[0m");

// data
uintmax_t nnz;
std::vector<uint32_t> rows, cols;
std::vector<double> vals_re, vals_im;
std::vector<int64_t> vals_int;

// reader
using mm_reader = matrix_market_reader<>;
mm_reader reader(&std::cout);

void read(const std::string& filename) 
{
    reader.open(filename);
    
    nnz = reader.nnz;
    rows.resize(nnz);
    cols.resize(nnz);

    switch (reader.type) {
        case mm_reader::INTEGER:
            vals_int.resize(nnz);
            break;

        case mm_reader::REAL:
            vals_re.resize(nnz);
            break;

        case mm_reader::COMPLEX:
            vals_re.resize(nnz);
            vals_im.resize(nnz);
            break;

        default: // prevents compiler warning
            break;
    }

    bool reverse_lexicographical_ordering = true;

    for (uintmax_t k = 0; k < nnz; k++) {
        uintmax_t row, col;
        double val_re, val_im;
        intmax_t val_int;
        reader.next_element(&row, &col, &val_re, &val_im, &val_int);

        rows[k] = row;
        cols[k] = col;

        switch (reader.type) {
            case mm_reader::INTEGER:
                vals_int[k] = val_int;
                break;

            case mm_reader::REAL:
                vals_re[k] = val_re;
                break;

            case mm_reader::COMPLEX:
                vals_re[k] = val_re;
                vals_im[k] = val_im;
                break;

            default: // prevents compiler warning
                break;
        }

        // check reverse lexicographical ordering
        if (k > 0) {
            if (cols[k] < cols[k - 1])
                reverse_lexicographical_ordering = false;
            if ((cols[k] == cols[k - 1]) && (rows[k] <= rows[k - 1]))
                reverse_lexicographical_ordering = false;
        }
    }

    std::cout << "Reverse lexicographical ordering: ";
    if (reverse_lexicographical_ordering)
        std::cout << green << "YES";
    else
        std::cout << red << "NO";
    std::cout << reset << std::endl;
}

void write(const std::string& filename)
{
    std::ofstream file(filename, std::ofstream::binary | std::ofstream::out);

    uint64_t m = reader.m;
    file.write((const char*)(&m), sizeof(uint64_t));
    uint64_t n = reader.n;
    file.write((const char*)(&n), sizeof(uint64_t));
    uint64_t nnz = reader.nnz;
    file.write((const char*)(&nnz), sizeof(uint64_t));
    uint64_t type = uint64_t(reader.type);
    file.write((const char*)(&type), sizeof(uint64_t));
    uint64_t hermit = uint64_t(reader.h);
    file.write((const char*)(&hermit), sizeof(uint64_t));

    file.write((const char*)rows.data(), sizeof(uint32_t) * rows.size());
    file.write((const char*)cols.data(), sizeof(uint32_t) * cols.size());

    switch (reader.type) {
        case mm_reader::INTEGER:
            file.write((const char*)vals_int.data(), sizeof(int64_t) * vals_int.size());
            break;

        case mm_reader::REAL:
            file.write((const char*)vals_re.data(), sizeof(double) * vals_re.size());
            break;

        case mm_reader::COMPLEX:
            file.write((const char*)vals_re.data(), sizeof(double) * vals_re.size());
            file.write((const char*)vals_im.data(), sizeof(double) * vals_im.size());
            break;

        default: // prevents compiler warning
            break;
    }
}

int main(int argc, char* argv[])
{
    // read matrix from .mtx file
    std::string filename(argv[1]);
    std::cout << "Reading matrix file: " << filename << std::endl;
    timer_type timer(timer_type::start_now);
    read(filename);
    timer.stop();
    std::cout << "Matrix reading time: " << yellow << timer.seconds() << reset << " [s]" << std::endl;

    // write matrix to .mm file
    filename = argv[2];
    std::cout << "Writing matrix file: " << filename << std::endl;
    timer.start();
    write(filename);
    timer.stop();
    std::cout << "Matrix reading time: " << cyan << timer.seconds() << reset << " [s]" << std::endl;
}
