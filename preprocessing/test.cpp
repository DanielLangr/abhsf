#include <omp.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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
uint64_t m, n, nnz;
enum Type { PATTERN, INTEGER, REAL, COMPLEX } type;
bool h; // if Hermitian (or symmetric)
std::vector<uint32_t> rows, cols;
std::vector<double> vals_re, vals_im;
std::vector<int64_t> vals_int;

void read_mtx(const std::string& filename) 
{
    matrix_market_reader<> reader;
    reader.open(filename);
    
    m = reader.m;
    n = reader.n;
    nnz = reader.nnz;
    type = (Type)((int)reader.type);
    h = reader.h;

    rows.resize(nnz);
    cols.resize(nnz);

    switch (type) {
        case INTEGER:
            vals_int.resize(nnz);
            break;

        case REAL:
            vals_re.resize(nnz);
            break;

        case COMPLEX:
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

        switch (type) {
            case INTEGER:
                vals_int[k] = val_int;
                break;

            case REAL:
                vals_re[k] = val_re;
                break;

            case COMPLEX:
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

void read_mm(const std::string& filename) 
{
    std::ifstream file(filename, std::ifstream::binary | std::ifstream::in);

    file.read((char*)&m, sizeof(uint64_t));
    file.read((char*)&n, sizeof(uint64_t));
    file.read((char*)&nnz, sizeof(uint64_t));

    uint64_t temp;
    file.read((char*)&temp, sizeof(uint64_t));
    type = (Type)temp;
    file.read((char*)&temp, sizeof(uint64_t));
    h = (bool)temp;

    rows.resize(nnz);
    file.read((char*)rows.data(), sizeof(uint32_t) * nnz);
    cols.resize(nnz);
    file.read((char*)cols.data(), sizeof(uint32_t) * nnz);

    switch (type) {
        case INTEGER:
            vals_int.resize(nnz);
            file.read((char*)vals_int.data(), sizeof(int64_t) * nnz);
            break;

        case REAL:
            vals_re.resize(nnz);
            file.read((char*)vals_re.data(), sizeof(double) * nnz);
            break;

        case COMPLEX:
            vals_re.resize(nnz);
            file.read((char*)vals_re.data(), sizeof(double) * nnz);
            vals_im.resize(nnz);
            file.read((char*)vals_im.data(), sizeof(double) * nnz);
            break;

        default: // prevents compiler warning
            break;
    }
}

template <typename Processor>
void iterate_bsk(const uintmax_t bsk, const int num_threads, const Processor& processor)
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

    timer_type timer(timer_type::start_now);
    switch (type) {
        case PATTERN:
            {
                auto swap_pat = [](std::size_t i, std::size_t j) {
                    std::swap(rows[i], rows[j]);
                    std::swap(cols[i], cols[j]);
                };

                #pragma omp parallel 
                aqsort::parallel_sort(nnz, &comp, &swap_pat);
            }
            break;

        case INTEGER:
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

        case REAL:
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

        case COMPLEX:
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
    timer.stop();
    std::cout << "Sorting time:     " << yellow
        << std::fixed << std::setprecision(4) << std::setw(8)
        << timer.seconds() << reset << " [s]" << std::endl;

    // iterate in parallel
    uintmax_t tb[num_threads + 1];
    tb[0] = 0;
    tb[num_threads] = nnz;

    omp_set_num_threads(num_threads);

    timer.start();
    #pragma omp parallel
    {
        int t = omp_get_thread_num();

        // find boundaries for threads
        if (t > 0) {
            uintmax_t l = (nnz * t) / num_threads;
            uintmax_t I = rows[l] >> bsk;
            uintmax_t J = cols[l] >> bsk;
            uintmax_t I_, J_;

            l++;
            while (l < nnz) {
                I_ = rows[l] >> bsk;
                J_ = cols[l] >> bsk;
                if ((I_ != I) || (J_ != J))
                    break;
                l++;
            };
            tb[t] = l;
        }

        #pragma omp barrier

        uintmax_t l1 = tb[t];
        uintmax_t I = rows[l1] >> bsk;
        uintmax_t J = cols[l1] >> bsk;

        for (uintmax_t l = tb[t] + 1; l < tb[t + 1]; l++) {
            uintmax_t I_ = rows[l] >> bsk;
            uintmax_t J_ = cols[l] >> bsk;
            if ((I_ != I) || (J_ != J)) {
                processor(l1, l - 1);
                l1 = l;
                I = I_;
                J = J_;
            }
        }
        if (l1 <= tb[t + 1] - 1)
            processor(l1, tb[t + 1] - 1);
    }
    timer.stop();
    std::cout << "Iteration time:   " << yellow
        << std::fixed << std::setprecision(4) << std::setw(8)
        << timer.seconds() << reset << " [s]" << std::endl;
}

template <typename Processor>
void iterate_s(const uintmax_t s, const int num_threads, const Processor& processor)
{
    // sort in parallel
    omp_set_num_threads(num_threads);

    auto comp = [s](std::size_t i, std::size_t j) {
        uintmax_t Ii = rows[i] / s;
        uintmax_t Ij = rows[j] / s;
        uintmax_t Ji = cols[i] / s;
        uintmax_t Jj = cols[j] / s;

        if (Ii < Ij) return true;
        if ((Ii == Ij) && (Ji < Jj)) return true;
        return false;
    };

    timer_type timer(timer_type::start_now);
    switch (type) {
        case PATTERN:
            {
                auto swap_pat = [](std::size_t i, std::size_t j) {
                    std::swap(rows[i], rows[j]);
                    std::swap(cols[i], cols[j]);
                };

                #pragma omp parallel 
                aqsort::parallel_sort(nnz, &comp, &swap_pat);
            }
            break;

        case INTEGER:
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

        case REAL:
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

        case COMPLEX:
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
    timer.stop();
    std::cout << "Sorting time:     " << yellow
        << std::fixed << std::setprecision(4) << std::setw(8)
        << timer.seconds() << reset << " [s]" << std::endl;

    // iterate in parallel
    uintmax_t tb[num_threads + 1];
    tb[0] = 0;
    tb[num_threads] = nnz;

    omp_set_num_threads(num_threads);

    timer.start();
    #pragma omp parallel
    {
        int t = omp_get_thread_num();

        // find boundaries for threads
        if (t > 0) {
            uintmax_t l = (nnz * t) / num_threads;
            uintmax_t I = rows[l] / s;
            uintmax_t J = cols[l] / s;
            uintmax_t I_, J_;

            l++;
            while (l < nnz) {
                I_ = rows[l] / s; 
                J_ = cols[l] / s;
                if ((I_ != I) || (J_ != J))
                    break;
                l++;
            };
            tb[t] = l;
        }

        #pragma omp barrier

        uintmax_t l1 = tb[t];
        uintmax_t I = rows[l1] / s;
        uintmax_t J = cols[l1] / s;

        for (uintmax_t l = tb[t] + 1; l < tb[t + 1]; l++) {
            uintmax_t I_ = rows[l] / s;
            uintmax_t J_ = cols[l] / s;
            if ((I_ != I) || (J_ != J)) {
                processor(l1, l - 1);
                l1 = l;
                I = I_;
                J = J_;
            }
        }
        if (l1 <= tb[t + 1] - 1)
            processor(l1, tb[t + 1] - 1);
    }
    timer.stop();
    std::cout << "Iteration time:   " << yellow
        << std::fixed << std::setprecision(4) << std::setw(8)
        << timer.seconds() << reset << " [s]" << std::endl;
}

void checksum()
{
    uint64_t c = 0;
    c += m;
    c += n;
    c += nnz;
    c += (uint64_t)type;
    c += (uint64_t)h;

    for (uintmax_t k = 0; k < nnz; k++) {
        c += (uint64_t)rows[k];
        c += (uint64_t)cols[k];
    }

    std::cout << "Checksum: " << std::hex << red << c << reset << std::dec << std::endl;
}

int main(int argc, char* argv[])
{
    // read matrix from file
    std::string filename(argv[1]);
    std::cout << "Reading matrix file: " << filename << std::endl;
    timer_type timer(timer_type::start_now);

    if (filename.substr(filename.size() - 4, 4) == ".mtx")
        read_mtx(filename);
    else if (filename.substr(filename.size() - 3, 3) == ".mm")
        read_mm(filename);
    else
        throw std::runtime_error("Unknown file format");
    timer.stop();
    std::cout << "Matrix reading time: " << yellow << timer.seconds() << reset << " [s]" << std::endl;

    uintmax_t s = std::atoi(argv[2]);
    uintmax_t bsk = 0;
    if (s == 0) {
        std::cout << "Block size: " << cyan << "2, 4, ..., 1024" << reset << std::endl;
        std::cout << "Block size power: " << cyan << "1, 2, ..., 10" << reset << std::endl;
    }
    else {
        // is s a power of 2?
        if ((s & (s - 1)) == 0) {
            // find exponent
            while ((s >> bsk) != 1)
                bsk++;
        }

        std::cout << "Block size: " << cyan << (1 << bsk) << reset <<std::endl;
        if (bsk > 0) 
            std::cout << "Block size power: " << cyan << bsk << reset << std::endl;
        else
            std::cout << "Block size power: " << red << "N/A" << reset << std::endl;
    }

    int num_threads = std::atoi(argv[3]);
    std::cout << "Number of threads: " << cyan << num_threads << reset << std::endl;

    // processor
    std::function<void(uintmax_t, uintmax_t)> processor;

    // nonzero-count processor data
    static std::atomic<uintmax_t> nnz_count {0};

    int processor_type = std::atoi(argv[4]);
    std::cout << "Processor type: ";
    if (processor_type == 0) {
        // nop (do nothing) processor
        std::cout << cyan << "do-nothing";
        processor = [](uintmax_t i1, uintmax_t i2){};
    }
    else if (processor_type == 1) {
        // chceck nonzeros processor
        std::cout << cyan << "nonzero-count";
        processor = [](uintmax_t i1, uintmax_t i2){ nnz_count += i2 - i1 + 1; };
    }
    std::cout << reset << std::endl;

    std::cout << "Matrix type: " << cyan << (m == n ? "SQUARE, " : "RECTANBULAR, ");
    switch (type) {
        case PATTERN:
            std::cout << "PATTERN, " << (h ? "SYMMETRIC" : "UNSYMMETRIC");
            break;

        case INTEGER:
            std::cout << "INTEGER, " << (h ? "SYMMETRIC" : "UNSYMMETRIC");
            break;

        case REAL:
            std::cout << "REAL, " << (h ? "SYMMETRIC" : "UNSYMMETRIC");
            break;

        case COMPLEX:
            std::cout << "COMPLEX, " << (h ? "HERMITIAN" : "UNSYMMETRIC");
            break;
    }
    std::cout << reset << std::endl;

    std::cout << "Number of rows:             " << cyan << std::right << std::setw(20) <<   m << reset << std::endl;
    std::cout << "Number of columns:          " << cyan << std::right << std::setw(20) <<   n << reset << std::endl;
    std::cout << "Number of nonzero elements: " << cyan << std::right << std::setw(20) << nnz << reset << std::endl;

    checksum();

    // iterations
    timer.start();

    if (s == 0) {
        for (bsk = 1; bsk <= 10; bsk++) {
            std::cout << "Tested block size: " << magenta << (1 << bsk) << reset << std::endl;
            iterate_bsk(bsk, num_threads, processor);
        }
    }
    else {
        if (bsk > 0)
            iterate_bsk(bsk, num_threads, processor);
        else 
            iterate_s(s, num_threads, processor);
    }
    timer.stop();

    std::cout << "Overall time:     " << green
        << std::fixed << std::setprecision(4) << std::setw(8)
        << timer.seconds() << reset << " [s]" << std::endl;

    if (processor_type == 1) {
        std::cout << "Nonzeros count check: ";
        if (nnz_count != nnz)
            std::cout << red << "FAILED";
        else
            std::cout << green << "PASSED";
        std::cout << reset << std::endl;
    }
}
