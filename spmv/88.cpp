#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

#include <immintrin.h>

#include <mkl.h>

#include <abhsf/utils/colors.h>
#include <abhsf/utils/timer.h>

using timer_type = chrono_timer<>;

//using real = float;
using real_type = double;

// block size fixed 8x8 for now
static const uint64_t block_size = 8;
static const uint64_t block_size_exp = 3;
static_assert(block_size == (1UL << block_size_exp), "Block size does not match its exponent.");

// number of blocks per block row / block column
// (num_blocks * num_blocks * block_size * block_size) elements must fit into memory
// (num_blocks * num_blocks) elements must not fit into cache
static const uint64_t num_blocks = 4096;

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

            std::uniform_int_distribution<> dis(0, block_size - 1);

            int k = 0;
            while (k < nnz) {
                uint32_t row = dis(gen_);
                uint32_t col = dis(gen_);

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
        block_coo_matrix_t()
            : brows_nnz_(nullptr), bcol_inds_(nullptr), bnnz_(nullptr), indices_(nullptr), values_(nullptr) { }
        ~block_coo_matrix_t() { free(brows_nnz_); free(bcol_inds_); free(bnnz_); free(indices_); free(values_); }

        void assemble(const block_generator& bgen)
        {
            free(brows_nnz_);
            if (posix_memalign((void**)(&brows_nnz_), 32, num_blocks * sizeof(uint16_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            free(bcol_inds_);
            if (posix_memalign((void**)(&bcol_inds_), 32, num_blocks * num_blocks * sizeof(uint16_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            free(bnnz_);
            if (posix_memalign((void**)(&bnnz_), 32, num_blocks * num_blocks * sizeof(uint8_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            const auto& nonzero_values = bgen.nonzero_values();
            const auto& indices = bgen.indices();
            assert(nonzero_values.size() == indices.size());

            free(indices_);
            if (posix_memalign((void**)(&indices_), 32,
                        num_blocks * num_blocks * indices.size() * sizeof(uint8_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            free(values_);
            if (posix_memalign((void**)(&values_), 32,
                        num_blocks * num_blocks * nonzero_values.size() * sizeof(real_type)) != 0)
                throw std::runtime_error("posix_memalign() error");

            uint16_t* brows_nnz_ptr = brows_nnz_;
            uint16_t* bcol_inds_ptr = bcol_inds_;
            uint8_t* bnnz_ptr = bnnz_;
            uint8_t* indices_ptr = indices_;
            real_type* values_ptr = values_;

            for (int brow = 0; brow < num_blocks; brow++) {
                *brows_nnz_ptr = num_blocks;
                brows_nnz_ptr++;

                for (int bcol = 0; bcol < num_blocks; bcol++) {
                    *bcol_inds_ptr = bcol;
                    bcol_inds_ptr++;

                    *bnnz_ptr = bgen.indices().size();
                    bnnz_ptr++;

                    memcpy(indices_ptr, indices.data(), indices.size() * sizeof(uint8_t));
                    indices_ptr += indices.size();

                    memcpy(values_ptr, nonzero_values.data(), nonzero_values.size() * sizeof(real_type));
                    values_ptr += nonzero_values.size();
                }
            }
        }

        void spmv(const real_type* __restrict__ x_, real_type* __restrict__ y_)
        {
            const real_type* __restrict__ x = (real_type*)__builtin_assume_aligned(x_, 32);
            real_type* __restrict__ y = (real_type*)__builtin_assume_aligned(y_, 32);
            uint16_t* __restrict__ brows_nnz = brows_nnz_; // (uint16_t*) __builtin_assume_aligned(brows_nnz_, 32);
            uint16_t* __restrict__ bcol_inds = bcol_inds_; // (uint16_t*) __builtin_assume_aligned(bcol_inds_, 32);
            uint8_t* __restrict__ bnnz = bnnz_;
            uint8_t* __restrict__ indices = indices_;
            real_type* __restrict__ values = values_; // (real_type*)__builtin_assume_aligned(values_, 32);

            for (uint32_t brow = 0; brow < num_blocks; brow++) {
                for (uint32_t j = 0; j < *brows_nnz; j++) {
                    const uint64_t bcol = *bcol_inds;
                    bcol_inds++;

                    const real_type* __restrict__ x_ptr =
                       (real_type*)__builtin_assume_aligned(x + bcol * block_size, 32);

                    for (uint32_t k = 0; k < *bnnz; k++) {
                        uint32_t lrow = *indices >> 4;
                        uint32_t lcol = *indices & 0x0F;

                        *(y + lrow) += *values * *(x_ptr + lcol);
                        values++;
                        indices++;
                    }
                    bnnz++;
                }

                brows_nnz++;
                y += block_size;
            }
        }

    private:
        uint16_t* __restrict__ brows_nnz_; // number of nonzero block per block rows
        uint16_t* __restrict__ bcol_inds_; // their block column indices
        uint8_t* __restrict__ bnnz_; // number of nonzero elements per each nonzero block
        uint8_t* __restrict__ indices_; // compressed row and column local in-block indices
        real_type* __restrict__ values_; // values of nonzero elements
};

class block_bitmap_matrix_t
{
    public:
        block_bitmap_matrix_t() : brows_nnz_(nullptr), bcol_inds_(nullptr), bitmaps_(nullptr), values_(nullptr) { }
        ~block_bitmap_matrix_t() { free(brows_nnz_); free(bcol_inds_); free(bitmaps_); free(values_); }

        void assemble(const block_generator& bgen)
        {
            free(brows_nnz_);
            if (posix_memalign((void**)(&brows_nnz_), 32, num_blocks * sizeof(uint16_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            free(bcol_inds_);
            if (posix_memalign((void**)(&bcol_inds_), 32, num_blocks * num_blocks * sizeof(uint16_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            free(bitmaps_);
            if (posix_memalign((void**)(&bitmaps_), 32, num_blocks * num_blocks * sizeof(uint64_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            const auto& nonzero_values = bgen.nonzero_values();
            assert(nonzero_values.size() == _mm_popcnt_u64(bgen.bitmap()));

            free(values_);
            if (posix_memalign((void**)(&values_), 32,
                        num_blocks * num_blocks * nonzero_values.size() * sizeof(real_type)) != 0)
                throw std::runtime_error("posix_memalign() error");

            uint16_t* brows_nnz_ptr = brows_nnz_;
            uint16_t* bcol_inds_ptr = bcol_inds_;
            uint64_t* bitmaps_ptr = bitmaps_;
            real_type* values_ptr = values_;

            for (int brow = 0; brow < num_blocks; brow++) {
                *brows_nnz_ptr = num_blocks;
                brows_nnz_ptr++;

                for (int bcol = 0; bcol < num_blocks; bcol++) {
                    *bcol_inds_ptr = bcol;
                    bcol_inds_ptr++;

                    *bitmaps_ptr = bgen.bitmap();
                    bitmaps_ptr++;

                    memcpy(values_ptr, nonzero_values.data(), nonzero_values.size() * sizeof(real_type));
                    values_ptr += nonzero_values.size();
                }
            }
        }

        void spmv(const real_type* __restrict__ x_, real_type* __restrict__ y_)
        {
            const real_type* __restrict__ x = (real_type*)__builtin_assume_aligned(x_, 32);
            real_type* __restrict__ y = (real_type*)__builtin_assume_aligned(y_, 32);
            uint16_t* __restrict__ brows_nnz = brows_nnz_; // (uint16_t*) __builtin_assume_aligned(brows_nnz_, 32);
            uint16_t* __restrict__ bcol_inds = bcol_inds_; // (uint16_t*) __builtin_assume_aligned(bcol_inds_, 32);
            uint64_t* __restrict__ bitmaps = bitmaps_;
            real_type* __restrict__ values = values_; // (real_type*)__builtin_assume_aligned(values_, 32);

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t j = 0; j < *brows_nnz; j++) {
                    const uint64_t bcol = *bcol_inds;
                    bcol_inds++;

                    uint64_t bitmap = *bitmaps;
                    bitmaps++;

                    const real_type* __restrict__ x_ptr =
                       (real_type*)__builtin_assume_aligned(x + bcol * block_size, 32);

                    static_assert(block_size == 8, "Block size expected to equal 8.");

                    const int64_t nnz = _mm_popcnt_u64(bitmap);
                    for (int64_t l = 0; l < nnz; l++) {
                     // const uint8_t k = _lzcnt_u64(bitmap);

                     // *(y + (k >> 3)) += *values * *(x_ptr + (k & 0x07));
                        *(y + 1) += *values * *(x_ptr + 1);
                        values++;

                        bitmap = _blsr_u64(bitmap);
                    }
/*
                    uint64_t k = _lzcnt_u64(bitmap);
                    while (k < 64) {
                        uint64_t lrow = k >> 3;
                        uint64_t lcol = k & 0x07;
                        *(y + lrow) += *values * *(x_ptr + lcol);
                        values++;

                        bitmap = _blsr_u64(bitmap);
                        k = _lzcnt_u64(bitmap);
                    }
*/
/*
                    for (uint64_t k = 0; k < 64; k++) {
                        uint64_t inc = bitmap & 0x01;
                        union { uint64_t i; double a; } temp;
                        temp.i = (~(inc - 1)) & 0x3FF0000000000000UL;
                        assert((temp.a == 0.0) || (temp.a == 1.0));

                        uint64_t lrow = k >> 3;
                        uint64_t lcol = k & 0x07;
                        *(y + lrow) += temp.a * *values * *(x_ptr + lcol);
                        values += inc;

                        bitmap >>= 1;
                    }
*/
/*
                    uint64_t k = 0;
                    while (bitmap != 0) {
                        if (bitmap & 0x01) {
                            uint64_t lrow = k >> 3;
                            uint64_t lcol = k & 0x07;
                            *(y + lrow) += *values * *(x_ptr + lcol);
                            values++;
                        }
                        bitmap >>= 1;
                        k++;
                    }
*/
/*
                    for (uint64_t k = 0; k < 64; k++) {
                        if (bitmap & 0x01) {
                            uint64_t lrow = k >> 3;
                            uint64_t lcol = k & 0x07;
                            *(y + lrow) += *values * *(x_ptr + lcol);
                            values++;
                        }
                        bitmap >>= 1;
                    }
*/
/*
                    for (uint64_t lrow = 0; lrow < 8; lrow++) {
                        for (uint64_t lcol = 0; lcol < 8; lcol++) {
                            if (bitmap & 0x01) {
                                *(y + lrow) += *values * *(x_ptr + lcol);
                                values++;
                            }
                            bitmap >>= 1;
                        }
                    }
*/
/*
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
*/
                }

                brows_nnz++;
                y += block_size;
            }
        }

        void spmv_2(
                const real_type* __restrict__ x_, real_type* __restrict__ y_)
        {
            const real_type* __restrict__ x = (real_type*)__builtin_assume_aligned(x_, 32);
            real_type* __restrict__ y = (real_type*)__builtin_assume_aligned(y_, 32);
            uint16_t* __restrict__ brows_nnz = brows_nnz_; // (uint16_t*) __builtin_assume_aligned(brows_nnz_, 32);
            uint16_t* __restrict__ bcol_inds = bcol_inds_; // (uint16_t*) __builtin_assume_aligned(bcol_inds_, 32);
            const uint64_t* __restrict__ bitmaps = bitmaps_;
            real_type* __restrict__ values = values_; // (real_type*)__builtin_assume_aligned(values_, 32);

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t j = 0; j < *brows_nnz; j++) {
                    const uint64_t bcol = *bcol_inds;
                    bcol_inds++;

                    uint64_t bitmap = *bitmaps;
                    bitmaps++;

                    const real_type* __restrict__ x_ptr =
                       (real_type*)__builtin_assume_aligned(x + bcol * block_size, 32);

                    static_assert(block_size == 8, "Block size expected to equal 8.");

                    for (uint64_t qrow = 0; qrow < 2; qrow++) {
                        real_type q_y[4];
                     // memcpy(q_y, y, 4 * sizeof(real_type));
                        for (int i = 0; i < 4; i++)
                            q_y[i] = *(y + i);

                        const real_type* __restrict__ q_x = x_ptr;
                        
                        real_type a[4];

                        // qcol = 0
                        uint16_t quad_bitmap = bitmap & 0xFFFF; 
                        bitmap >>= 16;

                     // if (quad_bitmap != 0) {
                        {
                            const uint16_t lcol_mask[] = { 0x000F, 0x00F0, 0x0F00, 0xF000 };
                            uint16_t lrow_mask[] = { 0x0001, 0x0002, 0x0004, 0x0008 };

                            for (int lcol = 0; lcol < 4; lcol++) {
                                if ((quad_bitmap & lcol_mask[lcol]) != 0) {
                                    for (int i = 0; i < 4; i++) {
                                        if ((quad_bitmap & lrow_mask[i]) == 0) {
                                            a[i] = 0.0;
                                        }
                                        else {
                                            a[i] = *values;
                                            values++;
                                        }
                                    }

                                    for (int i = 0; i < 4; i++)
                                        q_y[i] += a[i] * q_x[i];
                                }

                                for (int i = 0; i < 4; i++)
                                    lrow_mask[i] <<= 4;
                            }
                        }

                        // qcol = 1
                        quad_bitmap = bitmap & 0x0FFFF; 
                        bitmap >>= 16;

                     // if (quad_bitmap != 0) {
                        {
                            q_x += 4;

                            const uint16_t lcol_mask[] = { 0x000F, 0x00F0, 0x0F00, 0xF000 };
                            uint16_t lrow_mask[] = { 0x0001, 0x0002, 0x0004, 0x0008 };

                            for (int lcol = 0; lcol < 4; lcol++) {
                                if ((quad_bitmap & lcol_mask[lcol]) != 0) {
                                    for (int i = 0; i < 4; i++) {
                                        if ((quad_bitmap & lrow_mask[i]) == 0) {
                                            a[i] = 0.0;
                                        }
                                        else {
                                            a[i] = *values;
                                            values++;
                                        }
                                    }

                                    for (int i = 0; i < 4; i++)
                                        q_y[i] += a[i] * q_x[i];
                                }

                                for (int i = 0; i < 4; i++)
                                    lrow_mask[i] <<= 4;
                            }
                        }

                     // memcpy(y, q_y, 4 * sizeof(real_type));
                        for (int i = 0; i < 4; i++)
                            *(y + i) = q_y[i];
                        y += 4;
                    }
                    y -= 8;

/*
                    for (uint32_t lrow = 0; lrow < 8; lrow++) {
                        uint8_t row_bitmap = bitmap >> 8 * (7 - lrow);
                        if (row_bitmap > 0) {
                            for (int32_t lcol = 7; lcol >= 0; lcol--) {
                                if (((row_bitmap >> (7 - lcol)) & 1) == 1) {
                                    *(y + lrow) += *values * *(x_ptr + lcol);
                                    values++;
                                }
                            }
                        }
*/
                }

                brows_nnz++;
             // y += block_size;
            }
        }

    private:
        uint16_t* __restrict__ brows_nnz_; // same as for block_coo_matrix
        uint16_t* __restrict__ bcol_inds_; // same as for block_coo_matrix
        uint64_t* bitmaps_; // bitmap for each block in lexicographical order
        real_type* __restrict__ values_; // same as for block_coo_matrix
};

class block_dense_matrix_t
{
    public:
        block_dense_matrix_t() : brows_nnz_(nullptr), bcol_inds_(nullptr), values_(nullptr) { }
        ~block_dense_matrix_t() { free(brows_nnz_); free(bcol_inds_); free(values_); }

        void assemble(const block_generator& bgen)
        {
            free(brows_nnz_);
            if (posix_memalign((void**)(&brows_nnz_), 32, num_blocks * sizeof(uint16_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            free(bcol_inds_);
            if (posix_memalign((void**)(&bcol_inds_), 32, num_blocks * num_blocks * sizeof(uint16_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            free(values_);
            if (posix_memalign((void**)(&values_), 32,
                        num_blocks * num_blocks * block_size * block_size * sizeof(real_type)) != 0)
                throw std::runtime_error("posix_memalign() error");

            const auto& values = bgen.values();

            uint16_t* brows_nnz_ptr = brows_nnz_;
            uint16_t* bcol_inds_ptr = bcol_inds_;
            real_type* values_ptr = values_;

            for (int brow = 0; brow < num_blocks; brow++) {
                *brows_nnz_ptr = num_blocks;
                brows_nnz_ptr++;

                for (int bcol = 0; bcol < num_blocks; bcol++) {
                    *bcol_inds_ptr = bcol;
                    bcol_inds_ptr++;

                    memcpy(values_ptr, values.data(), values.size() * sizeof(real_type));
                    values_ptr += values.size();
                }
            }
        }

        void spmv(const real_type* __restrict__ x_, real_type* __restrict__ y_)
        {
            const real_type* __restrict__ x = (real_type*)__builtin_assume_aligned(x_, 32);
            real_type* __restrict__ y = (real_type*)__builtin_assume_aligned(y_, 32);
            uint16_t* __restrict__ brows_nnz = brows_nnz_; // (uint16_t*) __builtin_assume_aligned(brows_nnz_, 32);
            uint16_t* __restrict__ bcol_inds = bcol_inds_; // (uint16_t*) __builtin_assume_aligned(bcol_inds_, 32);
            real_type* __restrict__ values = values_; // (real_type*)__builtin_assume_aligned(values_, 32);

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t j = 0; j < *brows_nnz; j++) {
                    const uint64_t bcol = *bcol_inds;
                    bcol_inds++;

                    const real_type* __restrict__ x_ptr =
                       (real_type*)__builtin_assume_aligned(x + bcol * block_size, 32);

                    for (uint64_t lrow = 0; lrow < block_size; lrow++) 
                        for (uint64_t lcol = 0; lcol < block_size; lcol++) {
                            *(y + lrow) += *values * *(x_ptr + lcol);
                            values++;
                        }
                }

                brows_nnz++;
                y += block_size;
            }
        }
/*
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
*/
    private:
        uint16_t* __restrict__ brows_nnz_; // same as for block_coo_matrix
        uint16_t* __restrict__ bcol_inds_; // same as for block_coo_matrix
        real_type* __restrict__ values_; // values for both nonzero and zero elements
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

     // void spmv(const std::vector<real_type>& x, std::vector<real_type>& y)
        void spmv(const real_type* x, real_type* y)
        {
            for (MKL_INT row = 0; row < num_blocks * block_size; row++) 
                for (MKL_INT k = ia_[row]; k < ia_[row + 1]; k++)
                    y[row] += x[ja_[k]] * a_[k];
        }

        void spmv_mkl(const real_type* __restrict__ x, real_type* __restrict__ y)
        {
            const MKL_INT m = num_blocks * block_size;
         // const char transa = 'N';
         // const char metadescra[] = { 'G', ' ', ' ', 'C', ' ', ' ' };
            const double alpha = 1.0;
            
         // mkl_cspblas_dcsrgemv(&transa, &m, a_.data(), ia_.data(), ja_.data(), x, y);
            mkl_dcsrmv("N", &m, &m, &alpha, "G**C**",
                    a_.data(), ja_.data(), ia_.data(), ia_.data() + 1, x, &alpha, y);
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

    uint32_t const n = num_blocks * block_size;
 // real_type* x_ = (real_type*)aligned_alloc(32, n * sizeof(real_type));
 // real_type* y_ = (real_type*)aligned_alloc(32, n * sizeof(real_type));
    real_type *x, *y;
    posix_memalign((void**)(&x), 32, n * sizeof(real_type));
    posix_memalign((void**)(&y), 32, n * sizeof(real_type));
    for (uint64_t k = 0; k < n; k++) {
        x[k] = 1.0;
        y[k] = 0.0;
    }

    block_coo_matrix_t block_coo_matrix;
    block_bitmap_matrix_t block_bitmap_matrix;
    block_dense_matrix_t block_dense_matrix;
    csr_matrix_t csr_matrix;

 // for (int bnnz = 1; bnnz <= block_size * block_size; bnnz++) {
 // for (int bnnz = block_size * block_size; bnnz <= block_size * block_size; bnnz++) {
 // for (int bnnz = 1; bnnz <= block_size * block_size; bnnz *= 4) {
    for (int bnnz = 1; bnnz <= block_size * block_size; bnnz *= 2) {
 // for (int bnnz = 1; bnnz <= 1; bnnz++) {
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
                 // case block_dense_mkl: block_dense_matrix.spmv_mkl(x, y); break;
                    case csr:             csr_matrix.spmv(x, y);             break;
                    case csr_mkl:         csr_matrix.spmv_mkl(x, y);         break;
                    default: break;
                }
            }

            timer.stop();
            uint64_t n_flops = (uint64_t)2 * bnnz * num_blocks * num_blocks * num_iterations;
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
                << std::fixed << std::setprecision(2) << std::setw(10) << mflops_max << reset;

        real_type sum = 0.0;
        for (uint64_t k = 0; k < n; k++)
            sum += y[k];
        std::cout << "; y-vector sum: "
            << red << std::scientific << std::setprecision(15) << sum << reset << std::endl;

        for (uint64_t k = 0; k < n; k++) 
            y[k] = 0.0;
    }

    free(x);
    free(y);
}
