#include <omp.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
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

// data
uintmax_t nnz;
std::vector<uint32_t> rows, cols;
std::vector<double> vals_re, vals_im;
std::vector<intmax_t> vals_int;

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
    }
}

void iterate(const uintmax_t bsk, int num_threads)
{

    // sort in parallel
    omp_set_num_threads(num_threads);

    auto comp = [bsk](std::size_t i, std::size_t j) {
        uintmax_t Ii = rows[i] >> bsk;
        uintmax_t Ij = rows[j] >> bsk;
        uintmax_t Ji = cols[i] >> bsk;
        uintmax_t Jj = cols[j] >> bsk;

        if (Ii < Ij) return true;
        if ((Ii == Ij) && (Ji < Jj)) return true;
        return false;
    };

    switch (reader.type) {
        case mm_reader::PATTERN:
            {
                auto swap_pat = [](std::size_t i, std::size_t j) {
                    std::swap(rows[i], rows[j]);
                    std::swap(cols[i], cols[j]);
                };

                #pragma omp parallel 
                aqsort::parallel_sort(nnz, &comp, &swap_pat);
            }
            break;

        case mm_reader::INTEGER:
            {
                auto swap_int = [](std::size_t i, std::size_t j) {
                    std::swap(rows[i], rows[j]);
                    std::swap(cols[i], cols[j]);
                    std::swap(vals_int[i], vals_int[j]);
                };

                #pragma omp parallel 
                aqsort::parallel_sort(nnz, &comp, &swap_int);
            }
            break;

        case mm_reader::REAL:
            {
                auto swap_real = [](std::size_t i, std::size_t j) {
                    std::swap(rows[i], rows[j]);
                    std::swap(cols[i], cols[j]);
                    std::swap(vals_re[i], vals_re[j]);
                };

                #pragma omp parallel 
                aqsort::parallel_sort(nnz, &comp, &swap_real);
            }
            break;

        case mm_reader::COMPLEX:
            {
                auto swap_comp = [](std::size_t i, std::size_t j) {
                    std::swap(rows[i], rows[j]);
                    std::swap(cols[i], cols[j]);
                    std::swap(vals_re[i], vals_re[j]);
                    std::swap(vals_im[i], vals_im[j]);
                };

                #pragma omp parallel 
                aqsort::parallel_sort(nnz, &comp, &swap_comp);
            }
            break;
    }

    // iterate in parallel
    uintmax_t tb[num_threads + 1];
    tb[0] = 0;
    tb[num_threads] = nnz;
    const uintmax_t s = 1UL << bsk;

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int t = omp_get_thread_num();

        // find boundaries for threads
        if (t > 0) {
            uintmax_t l = (nnz * t) / num_threads;
            uintmax_t I = rows[l] >> bsk;
            uintmax_t J = cols[l] >> bsk;
            uintmax_t I_, J_;

            do {
                l++;
                I_ = rows[l] >> bsk;
                J_ = cols[l] >> bsk;
            } while ((l < (nnz - 1)) && (I_ == I) && (J_ == J));
            tb[t] = l;
        }

        #pragma omp barrier

        uintmax_t l1 = tb[t];
        uintmax_t I = rows[l1] >> bsk;
        uintmax_t J = cols[l1] >> bsk;

        for (uintmax_t l = tb[t]; l < tb[t + 1] - 1; l++) {
            uintmax_t I_ = rows[l] >> bsk;
            uintmax_t J_ = cols[l] >> bsk;
            if ((I_ != I) || (J_ != J)) {
                // process block with indexes between l1 and (l - 1) inclusive

                l1 = l;
                I = I_;
                J = J_;
            }
        }

        // process block with indexes between l1 and (tb[t + 1] - 1) inclusive
    }
}

int main(int argc, char* argv[])
{
    // read matrix from file
    std::string filename(argv[1]);
    std::cout << "Reading matrix file: " << filename << std::endl;
    timer_type timer(timer_type::start_now);
    read(filename);
    timer.stop();
    std::cout << "Matrix reading time: " << yellow << timer.seconds() << reset << " [s]" << std::endl;

    const uintmax_t bsk = std::atoi(argv[2]);
    std::cout << "Block size power: " << cyan << bsk << reset << std::endl;
    std::cout << "Block size: " << cyan << (1 << bsk) << reset <<std::endl;

    int num_threads = std::atoi(argv[3]);
    std::cout << "Number of threads: " << cyan << num_threads << reset << std::endl;

    timer.start();
    iterate(bsk, num_threads);
    timer.stop();
    std::cout << "Iteration over blocks: " << green << timer.seconds() << reset << std::endl;
}
