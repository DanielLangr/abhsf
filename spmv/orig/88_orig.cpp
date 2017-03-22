#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <mkl.h>

#include <abhsf/utils/colors.h>
#include <abhsf/utils/timer.h>

using timer_type = chrono_timer<>;

//using real = float;
using real_type = double;

// block size fixed 8x8 for now
static const uint32_t block_size = 8;
static const uint32_t block_size_exp = 3;
static_assert(block_size == (1UL << block_size_exp), "Block size does not match its exponent.");

// number of blocks per block row / block column
// (num_blocks * num_blocks * block_size * block_size) elements must fit into memory
// (num_blocks * num_blocks) elements must not fit into cache
static const uint32_t num_blocks = 4096;

static const int num_experiments = 3;
static const int num_iterations = 20;
static const int warm_up_iterations = 5;

class block_generator
{
    public:
        block_generator() : gen_(0x0498bc04) { } // same seed to generate same "random" blocks 

        void generate_random(int nnz)
        {
            assert((nnz > 0) && (nnz <= block_size * block_size));

            set_.clear();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, block_size - 1);

            int k = 0;
            while (k < nnz) {
                uint32_t row = dis(gen);
                uint32_t col = dis(gen);

                if (set_.find(std::make_pair(row, col)) == set_.end()) {
                    set_.emplace(row, col);
                    k++;
                }
            }

            finalize();
        }

        void generate_diagonal()
        {
            set_.clear();
            for (int k = 0; k < block_size; k++)
                set_.emplace(k, k);

            finalize();
        }

        void generate_single_row()
        {
            set_.clear();
            for (int k = 0; k < block_size; k++)
                set_.emplace(0, k);

            finalize();
        }

        void generate_single_col()
        {
            set_.clear();
            for (int k = 0; k < block_size; k++)
                set_.emplace(k, 0);

            finalize();
        }

        const std::vector<real_type>& nonzero_values() const { return nonzero_values_; }
        const std::vector<real_type>& values() const { return values_; }
        const std::vector<uint8_t>& indices() const { return indices_; }
        uint64_t bitmap() const { return bitmap_; }

    private:
        void finalize()
        {
            nonzero_values_.clear();
            values_.clear();
            indices_.clear();
            bitmap_ = 0;

            auto iter = set_.cbegin();

            for (int row = 0; row < block_size; row++) {
                for (int col = 0; col < block_size; col++) {
                    if ((iter->first == row) && (iter->second == col)) {
                        nonzero_values_.emplace_back(1.0);
                        values_.emplace_back(1.0);

                        static_assert(block_size_exp < 4, "Block size expected to be less or equal than 16.");
                        indices_.emplace_back((row << 4) + col);

                        static_assert(block_size == 8, "Block size expected to equal 8.");
                        bitmap_ |= 1UL << ((8 * row) + col);

                        ++iter;
                    }
                    else 
                        values_.emplace_back(0.0);
                }
            }
        }

        std::set<std::pair<uint32_t, uint32_t>> set_;

        std::vector<real_type> nonzero_values_;
        std::vector<real_type> values_;

        std::vector<uint8_t> indices_; // row and column indices
        uint64_t bitmap_; // bitmap describing block structure

        std::mt19937 gen_;
};

class block_coo_matrix_t
{
    public:
        void assemble(const block_generator& bgen)
        {
            brows_nnz_.clear();
            bcol_inds_.clear();
            values_.clear();
            bnnz_.clear();
            indices_.clear();

            const auto& nonzero_values = bgen.nonzero_values();
            const auto& indices = bgen.indices();

            brows_nnz_.reserve(num_blocks);
            bcol_inds_.reserve(num_blocks * num_blocks);
            values_.reserve(num_blocks * num_blocks * nonzero_values.size());
            bnnz_.reserve(num_blocks * num_blocks);
            indices_.reserve(num_blocks * num_blocks * indices.size());

            for (int brow = 0; brow < num_blocks; brow++) {
                brows_nnz_.emplace_back(num_blocks);

                for (int bcol = 0; bcol < num_blocks; bcol++) {
                    bcol_inds_.emplace_back(bcol);

                    values_.insert(values_.end(), nonzero_values.cbegin(), nonzero_values.cend());
                    bnnz_.emplace_back(bgen.indices().size());
                    indices_.insert(indices_.end(), indices.cbegin(), indices.cend());
                }
            }
        }

        void spmv(const std::vector<real_type>& x, std::vector<real_type>& y)
        {
            auto bcol_iter = bcol_inds_.cbegin();
            auto values_iter = values_.cbegin();
            auto bnnz_iter = bnnz_.cbegin();
            auto indices_iter = indices_.cbegin();

            // optimizations? vectorization, software prefetching, ...

            for (uint32_t brow = 0; brow < num_blocks; brow++) {
                for (uint32_t j = 0; j < brows_nnz_[brow]; j++) {
                    uint32_t bcol = *bcol_iter;

                    for (uint32_t k = 0; k < *bnnz_iter; k++) {
                        uint32_t lrow = *indices_iter >> 4;
                        uint32_t lcol = *indices_iter & 0x0F;

                        y[brow * block_size + lrow] += *values_iter * x[bcol * block_size + lcol];

                        ++values_iter;
                        ++indices_iter;
                    }
                    ++bcol_iter;
                    ++bnnz_iter;
                }
            }

        }

    private:
        std::vector<uint16_t> brows_nnz_; // number of nonzero block per block rows
        std::vector<uint16_t> bcol_inds_; // their block column indices
        std::vector<real_type> values_; // values of nonzero elements
        std::vector<uint8_t> bnnz_; // number of nonzero elements per each nonzero block
        std::vector<uint8_t> indices_; // compressed row and column local in-block inidces
};

class block_bitmap_matrix_t
{
    public:
        void assemble(const block_generator& bgen)
        {
            brows_nnz_.clear();
            bcol_inds_.clear();
            values_.clear();
            bitmap_.clear();

            const auto& nonzero_values = bgen.nonzero_values();

            brows_nnz_.reserve(num_blocks);
            bcol_inds_.reserve(num_blocks * num_blocks);
            values_.reserve(num_blocks * num_blocks * nonzero_values.size());
            bitmap_.reserve(num_blocks * num_blocks);

            for (int brow = 0; brow < num_blocks; brow++) {
                brows_nnz_.emplace_back(num_blocks);

                for (int col = 0; col < num_blocks; col++) {
                    bcol_inds_.emplace_back(col);

                    values_.insert(values_.end(), nonzero_values.cbegin(), nonzero_values.cend());
                    bitmap_.emplace_back(bgen.bitmap());
                }
            }
        }

        void spmv(const std::vector<real_type>& x, std::vector<real_type>& y)
        {
            auto bcol_iter = bcol_inds_.cbegin();
            auto values_iter = values_.cbegin();
            auto bitmap_iter = bitmap_.cbegin();

            // optimizations? vectorization, software prefetching, ...

            for (uint32_t brow = 0; brow < num_blocks; brow++) {
                for (uint32_t j = 0; j < brows_nnz_[brow]; j++) {
                    uint32_t bcol = *bcol_iter;
                    uint64_t bitmap = *bitmap_iter;

                    static_assert(block_size == 8, "Block size expected to equal 8.");
                    for (uint32_t lrow = 0; lrow < 8; lrow++) {
                        uint8_t row_bitmap = bitmap >> 8 * (7 - lrow);
                        if (row_bitmap > 0) {
                            for (int32_t lcol = 7; lcol >= 0; lcol--) {
                                if (((row_bitmap >> (7 - lcol)) & 1) == 1) {
                                    y[brow * 8 + lrow] += *values_iter * x[bcol * 8 + lcol];
                                    ++values_iter;
                                }
                            }
                        }
                    }
                    
                    ++bcol_iter;
                    ++bitmap_iter;
                }
            }

        }

    private:
        std::vector<uint16_t> brows_nnz_; // same as for block_coo_matrix
        std::vector<uint16_t> bcol_inds_; // same as for block_coo_matrix
        std::vector<real_type> values_; // same as for block_coo_matrix
        std::vector<uint64_t> bitmap_; // bitmap for each block in lexicographical order
};

class block_dense_matrix_t
{
    public:
        void assemble(const block_generator& bgen)
        {
            brows_nnz_.clear();
            bcol_inds_.clear();
            values_.clear();

            const auto& values = bgen.values();

            brows_nnz_.reserve(num_blocks);
            bcol_inds_.reserve(num_blocks * num_blocks);
            values_.reserve(num_blocks * num_blocks * block_size * block_size);

            for (int brow = 0; brow < num_blocks; brow++) {
                brows_nnz_.emplace_back(num_blocks);

                for (int bcol = 0; bcol < num_blocks; bcol++) {
                    bcol_inds_.emplace_back(bcol);
                    values_.insert(values_.end(), values.cbegin(), values.cend());
                }
            }
        }

        void spmv(const std::vector<real_type>& x, std::vector<real_type>& y)
        {
            auto bcol_iter = bcol_inds_.cbegin();
            auto values_iter = values_.cbegin();

            for (uint32_t brow = 0; brow < num_blocks; brow++) {
                for (uint32_t j = 0; j < brows_nnz_[brow]; j++) {
                    uint32_t bcol = *bcol_iter;

                    for (uint32_t lrow = 0; lrow < block_size; lrow++) {
                        for (uint32_t lcol = 0; lcol < block_size; lcol++) {
                            y[brow * block_size + lrow] += *values_iter * x[bcol * block_size + lcol];
                            ++values_iter;
                        }
                    }
                    
                    ++bcol_iter;
                }
            }
        }

        void spmv_mkl(const std::vector<real_type>& x, std::vector<real_type>& y)
        {
            auto bcol_iter = bcol_inds_.cbegin();
            real_type* a = values_.data();

            for (uint32_t brow = 0; brow < num_blocks; brow++) {
                for (uint32_t j = 0; j < brows_nnz_[brow]; j++) {
                    uint32_t bcol = *bcol_iter;

                    cblas_dgemv(CblasRowMajor, CblasNoTrans, block_size, block_size, 1.0, a, block_size, 
                           x.data() + bcol * block_size, 1, 0.0, y.data() + brow * block_size, 1); 

                    a += block_size * block_size;
                    ++bcol_iter;
                }
           }
        }

    private:
        std::vector<uint16_t> brows_nnz_; // same as for block_coo_matrix
        std::vector<uint16_t> bcol_inds_; // same as for block_coo_matrix
        std::vector<real_type> values_; // values for both nonzero and zero elements
};

// non-blocking CSR matrix with 32-bit indices
class csr_matrix_t
{
    public:
        void assemble(const block_generator& bgen)
        {
            a_.clear();
            ia_.clear();
            ja_.clear();

            const auto& nonzero_values = bgen.nonzero_values();
            const auto& indices = bgen.indices();

            a_.reserve(num_blocks * num_blocks * nonzero_values.size());
            ia_.reserve(num_blocks * block_size + 1);
            ja_.reserve(num_blocks * num_blocks * nonzero_values.size());

            using element_t = std::tuple<uint32_t, uint32_t, real_type>;
            std::vector<element_t> elements;
            uintmax_t nnz = num_blocks * num_blocks * nonzero_values.size();
            elements.reserve(nnz);

            for (uint32_t brow = 0; brow < num_blocks; brow++) {
                for (uint32_t bcol = 0; bcol < num_blocks; bcol++) {
                    for (int k = 0; k < nonzero_values.size(); k++) {
                        static_assert(block_size_exp < 4, "Block size expected to be less or equal than 16.");
                        elements.emplace_back(
                                brow * block_size + (indices[k] >> 4),
                                bcol * block_size + (indices[k] & 0x0F),
                                nonzero_values[k]);
                    }
                }
            }

            // to-CSR conversion
            std::sort(elements.begin(), elements.end());

            for (size_t k = 0; k < nnz; k++) {
                a_[k] = std::get<2>(elements[k]);
                ja_[k] = std::get<1>(elements[k]);
            }

            ia_[0] = 0;
            size_t k = 0;
            size_t row = 0;

            while (k < nnz) {
                while ((k < nnz) && (row == std::get<0>(elements[k])))
                    k++;

                row++;
                ia_[row] = k;
            }

        }

        void spmv(const std::vector<real_type>& x, std::vector<real_type>& y)
        {
            for (MKL_INT row = 0; row < num_blocks * block_size; row++) 
                for (MKL_INT k = ia_[row]; k < ia_[row + 1]; k++)
                    y[row] += x[ja_[k]] * a_[k];
        }

        void spmv_mkl(const std::vector<real_type>& x, std::vector<real_type>& y)
        {
            const MKL_INT m = num_blocks * block_size;
            const char transa = 'N';
            
            mkl_cspblas_dcsrgemv(&transa, &m, a_.data(), ia_.data(), ja_.data(), x.data(), y.data());
        }

    private:
        std::vector<real_type> a_;

        static_assert(sizeof(MKL_INT) == 4, "Size of MKL_INT must be 4");

        std::vector<MKL_INT> ia_;
        std::vector<MKL_INT> ja_;
};

int main(int argc, char* argv[])
{
    std::cout << "Blocks size: " << magenta << block_size << reset << std::endl;
    std::cout << "Number of block rows/columns: " << magenta << num_blocks << reset << std::endl;
    std::cout << "Number of random experiments: " << magenta << num_experiments << reset << std::endl;
    std::cout << "Number of iterations: " << magenta << num_iterations << reset << std::endl;
    std::cout << "Number of warm-up iterations: " << magenta << warm_up_iterations << reset << std::endl;

    // types of matrices / spmv variants
    enum type_t {
        block_coo,
        block_bitmap,
        block_dense,
        block_dense_mkl,
        csr,
        csr_mkl
    } type = block_coo;

    if (argc > 1)
        type = static_cast<type_t>(atol(argv[1]));

    std::cout << "Matrix / SpMV type: " << yellow;
    switch (type) {
        case block_coo:       std::cout << "BLOCK COO";       break;
        case block_bitmap:    std::cout << "BLOCK BITMAP";    break;
        case block_dense:     std::cout << "BLOCK DENSE";     break;
        case block_dense_mkl: std::cout << "BLOCK DENSE MKL"; break;
        case csr:             std::cout << "CSR";             break;
        case csr_mkl:         std::cout << "CSR MKL";         break;
        default:              throw std::runtime_error("Unknown matrix/spmv type!");
    }
    std::cout << reset << std::endl;

    block_generator bgen;

    std::vector<real_type> x(num_blocks * block_size, 1.0);
    std::vector<real_type> y(num_blocks * block_size, 0.0);

    block_coo_matrix_t block_coo_matrix;
    block_bitmap_matrix_t block_bitmap_matrix;
    block_dense_matrix_t block_dense_matrix;
    csr_matrix_t csr_matrix;

 // for (int bnnz = 1; bnnz <= block_size * block_size; bnnz++) {
    for (int bnnz = block_size * block_size; bnnz <= block_size * block_size; bnnz++) {
        double mflops_min, mflops_max, mflops_avg, mflops_sum = 0.0;

        for (int exp = 0; exp < num_experiments; exp++) {
            bgen.generate_random(bnnz);

            switch (type) {
                case block_coo:       block_coo_matrix.assemble(bgen);    break;
                case block_bitmap:    block_bitmap_matrix.assemble(bgen); break;
                case block_dense:
                case block_dense_mkl: block_dense_matrix.assemble(bgen);  break;
                case csr:
                case csr_mkl:         csr_matrix.assemble(bgen);          break;
                default: break;
            }

            timer_type timer;

            for (int iter = -warm_up_iterations; iter < num_iterations; iter++) {
                if (iter == 0)
                    timer.start();

                switch (type) {
                    case block_coo:       block_coo_matrix.spmv(x, y);       break;
                    case block_bitmap:    block_bitmap_matrix.spmv(x, y);    break;
                    case block_dense:     block_dense_matrix.spmv(x, y);     break;
                    case block_dense_mkl: block_dense_matrix.spmv_mkl(x, y); break;
                    case csr:             csr_matrix.spmv(x, y);             break;
                    case csr_mkl:         csr_matrix.spmv_mkl(x, y);         break;
                    default: break;
                }
            }

            timer.stop();
            uintmax_t n_flops = (uintmax_t)2 * bnnz * num_blocks * num_blocks * num_iterations;
            double mflops = double(n_flops) / 1.0e6 / timer.seconds();

            if (exp == 0) 
                mflops_min = mflops_max = mflops_sum = mflops;
            else {
                mflops_min = std::min(mflops_min, mflops);
                mflops_max = std::max(mflops_max, mflops);
                mflops_sum += mflops;
            }
        }

        mflops_avg = mflops_sum / double(num_experiments);

        std::cout
            << "Block nonzeros = " << cyan << std::right << std::setw(4) << bnnz << reset
            << ", MFLOP/s: min = " << green << std::right
                << std::fixed << std::setprecision(2) << std::setw(10) << mflops_min << reset
            << ", avg = " << green << std::right
                << std::fixed << std::setprecision(2) << std::setw(10) << mflops_avg << reset
            << ", max = " << green << std::right
                << std::fixed << std::setprecision(2) << std::setw(10) << mflops_max << reset
            << std::endl;
    }

    std::cout << "y-vector sum: " << red << std::scientific << std::setprecision(15) 
        << std::accumulate(y.cbegin(), y.cend(), 0.0) << reset << std::endl;
}
