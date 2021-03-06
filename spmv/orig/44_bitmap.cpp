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

#include <abhsf/utils/colors.h>
#include <abhsf/utils/timer.h>

#ifdef __INTEL_COMPILER
    #define RESTRICT restrict
#else
    #define RESTRICT __restrict__
#endif

using timer_type = chrono_timer<>;

//using real_type = float;
using real_type = double;

// block size fixed 8x8 for now
static const uint64_t block_size = 4;
static const uint64_t block_size_exp = 2;
static_assert(block_size == (1UL << block_size_exp), "Block size does not match its exponent.");

// number of blocks per block row / block column
// (num_blocks * num_blocks * block_size * block_size) elements must fit into memory
// (num_blocks * num_blocks) elements must not fit into cache
static const uint64_t num_blocks = 8192;

static const int num_experiments = 5;
static const int num_iterations = 8;
static const int warm_up_iterations = 2;

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
/*
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
*/
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

                        static_assert(block_size == 4, "Block size expected to equal 4.");
                        bitmap_ |= 1UL << ((4 * row) + col);

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
        uint16_t bitmap_; // bitmap describing block structure

        std::mt19937 gen_;
};

class block_bitmap_matrix_t
{
    public:
        block_bitmap_matrix_t() : bitmaps_(nullptr), values_(nullptr)
        {
            static const uint64_t m = 0xFFFFFFFFFFFFFFFFUL;
            mask_lookup[0] = _mm256_set_epi64x(0, 0, 0, 0);
            mask_lookup[1] = _mm256_set_epi64x(0, 0, 0, m);
            mask_lookup[2] = _mm256_set_epi64x(0, 0, m, 0);
            mask_lookup[3] = _mm256_set_epi64x(0, 0, m, m);
            mask_lookup[4] = _mm256_set_epi64x(0, m, 0, 0);
            mask_lookup[5] = _mm256_set_epi64x(0, m, 0, m);
            mask_lookup[6] = _mm256_set_epi64x(0, m, m, 0);
            mask_lookup[7] = _mm256_set_epi64x(0, m, m, m);
            mask_lookup[8] = _mm256_set_epi64x(m, 0, 0, 0);
            mask_lookup[9] = _mm256_set_epi64x(m, 0, 0, m);
            mask_lookup[10] = _mm256_set_epi64x(m, 0, m, 0);
            mask_lookup[11] = _mm256_set_epi64x(m, 0, m, m);
            mask_lookup[12] = _mm256_set_epi64x(m, m, 0, 0);
            mask_lookup[13] = _mm256_set_epi64x(m, m, 0, m);
            mask_lookup[14] = _mm256_set_epi64x(m, m, m, 0);
            mask_lookup[15] = _mm256_set_epi64x(m, m, m, m);

            perm_lookup[0] = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            perm_lookup[1] = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            perm_lookup[2] = _mm256_set_epi32(7, 6, 5, 4, 1, 0, 3, 2);
            perm_lookup[3] = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            perm_lookup[4] = _mm256_set_epi32(7, 6, 1, 0, 5, 4, 3, 2);
            perm_lookup[5] = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
            perm_lookup[6] = _mm256_set_epi32(7, 6, 3, 2, 1, 0, 5, 4);
            perm_lookup[7] = _mm256_set_epi32(7, 6, 3, 2, 1, 0, 5, 4);
            perm_lookup[8] = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
            perm_lookup[9] = _mm256_set_epi32(3, 2, 7, 6, 5, 4, 1, 0);
            perm_lookup[10] = _mm256_set_epi32(3, 2, 7, 6, 1, 0, 5, 4);
            perm_lookup[11] = _mm256_set_epi32(5, 4, 7, 6, 3, 2, 1, 0);
            perm_lookup[12] = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6);
            perm_lookup[13] = _mm256_set_epi32(5, 4, 3, 2, 7, 6, 1, 0);
            perm_lookup[14] = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6);
            perm_lookup[15] = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        }

        ~block_bitmap_matrix_t() { free(bitmaps_); free(values_); }

        void assemble(const block_generator& bgen)
        {
            free(bitmaps_);
            if (posix_memalign((void**)(&bitmaps_), 64, num_blocks * num_blocks * sizeof(uint16_t)) != 0)
                throw std::runtime_error("posix_memalign() error");

            const auto& nonzero_values = bgen.nonzero_values();
            assert(nonzero_values.size() == _mm_popcnt_u64(bgen.bitmap()));

            free(values_);
            if (posix_memalign((void**)(&values_), 64,
                     // num_blocks * num_blocks * nonzero_values.size() * sizeof(real_type)) != 0)
                        (num_blocks * num_blocks * nonzero_values.size() + 8) * sizeof(real_type)) != 0)
                throw std::runtime_error("posix_memalign() error");

            uint16_t* bitmaps_ptr = bitmaps_;
            real_type* values_ptr = values_;

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    *bitmaps_ptr = bgen.bitmap();
                    bitmaps_ptr++;

                    memcpy(values_ptr, nonzero_values.data(), nonzero_values.size() * sizeof(real_type));
                    values_ptr += nonzero_values.size();
                }
            }
        }

        void spmv_basic(const real_type* x, real_type* y)
        {
            uint16_t* bitmaps = bitmaps_;
            real_type* values = values_;

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    const real_type* x_ptr = x + (bcol << block_size_exp);

                    uint16_t bitmap = *bitmaps;
                    bitmaps++;

                    for (uint64_t lrow = 0; lrow < block_size; lrow++) {
                        for (uint64_t lcol = 0; lcol < block_size; lcol++) {
                            if (bitmap & 0x01) {
                                *(y + lrow) += *values * *(x_ptr + lcol);
                                values++;
                            }
                            bitmap >>= 1;
                        }
                    }
                }
                y += block_size;
            }
        }

        void spmv_basic_row(const real_type* x, real_type* y)
        {
            uint16_t* bitmaps = bitmaps_;
            real_type* values = values_;

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    const real_type* x_ptr = x + (bcol << block_size_exp);

                    uint16_t block_bitmap = *bitmaps;
                    bitmaps++;

                    for (uint64_t lrow = 0; lrow < block_size; lrow++) {
                        uint16_t row_bitmap = block_bitmap & 0x0F;
                        block_bitmap >>= 4;

                        if (row_bitmap > 0) {
                            for (uint64_t lcol = 0; lcol < block_size; lcol++) {
                                if (row_bitmap & 0x01) {
                                    *(y + lrow) += *values * *(x_ptr + lcol);
                                    values++;
                                }
                                row_bitmap >>= 1;
                            }
                        }
                    }
                }
                y += block_size;
            }
        }

        void spmv_basic_row_2(const real_type* x, real_type* y)
        {
            uint16_t* bitmaps = bitmaps_;
            real_type* values = values_;

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    const real_type* x_ptr = x + (bcol << block_size_exp);

                    uint16_t block_bitmap = *bitmaps;
                    bitmaps++;

                    for (uint64_t lrow = 0; lrow < block_size; lrow++) {
                        uint16_t row_bitmap = (block_bitmap >> (lrow << 2)) & 0x0F;

                        if (row_bitmap > 0) {
                            for (uint64_t lcol = 0; lcol < block_size; lcol++) {
                                if (row_bitmap & 0x01) {
                                    *(y + lrow) += *values * *(x_ptr + lcol);
                                    values++;
                                }
                                row_bitmap >>= 1;
                            }
                        }
                    }
                }
                y += block_size;
            }
        }

        void spmv_popcnt_lzcnt_blsr(const real_type* x, real_type* y)
        {
            uint16_t* bitmaps = bitmaps_;
            real_type* values = values_;

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    uint64_t bitmap = (uint64_t)(*bitmaps);
                    bitmaps++;

                    const real_type* x_ptr = x + (bcol << block_size_exp);

                    const int64_t nnz = _mm_popcnt_u64(bitmap); // number of set bits
                    for (int64_t l = 0; l < nnz; l++) {
                        const uint8_t k = _lzcnt_u64(bitmap); // number of leading zero bits
                        *(y + (k >> 2)) += *values * *(x_ptr + (k & 0x03));
                        values++;
                        bitmap = _blsr_u64(bitmap); // reset lowest set bit
                    }
                }
                y += block_size;
            }
        }

        void spmv_temp(const real_type* x, real_type* y)
        {

            uint16_t* bitmaps = bitmaps_;
            real_type* values = values_;

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                __m256d ry0 = _mm256_setzero_pd();
                __m256d ry1 = _mm256_setzero_pd();
                __m256d ry2 = _mm256_setzero_pd();
                __m256d ry3 = _mm256_setzero_pd();

                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    uint16_t bitmap = *bitmaps;
                    bitmaps++;

                    const real_type* x_ptr = x + (bcol << block_size_exp);

                    __m256d rx0 = _mm256_load_pd(x_ptr);
                    __m256d rx1 = _mm256_permute4x64_pd(rx0, 0b00111001);
                    __m256d rx2 = _mm256_permute4x64_pd(rx0, 0b01001110);
                    __m256d rx3 = _mm256_permute4x64_pd(rx0, 0b10010011);

                    uint8_t diag_bitmap_0 = bitmap & 0x0F;
                    uint8_t diag_bitmap_1 = (bitmap >> 4) & 0x0F;
                    uint8_t diag_bitmap_2 = (bitmap >> 8) & 0x0F;
                    uint8_t diag_bitmap_3 = (bitmap >> 12) & 0x0F;

                    uint8_t diag_nnz_0 = popcnt_lookup[diag_bitmap_0];
                    uint8_t diag_nnz_1 = popcnt_lookup[diag_bitmap_1];
                    uint8_t diag_nnz_2 = popcnt_lookup[diag_bitmap_2];
                    uint8_t diag_nnz_3 = popcnt_lookup[diag_bitmap_3];

                    uint8_t pp_1 = diag_nnz_0;
                    uint8_t pp_2 = diag_nnz_0 + diag_nnz_1;
                    uint8_t pp_3 = diag_nnz_2;
                    uint8_t pp_4 = diag_nnz_2 + diag_nnz_3;;
                    pp_3 += pp_2;
                    pp_4 += pp_2;

                    __m256d rd0 = _mm256_maskload_pd(values, mask_lookup[diag_bitmap_0]);
                    __m256d rd1 = _mm256_maskload_pd(values + pp_1, mask_lookup[diag_bitmap_1]);
                    __m256d rd2 = _mm256_maskload_pd(values + pp_2, mask_lookup[diag_bitmap_2]);
                    __m256d rd3 = _mm256_maskload_pd(values + pp_3, mask_lookup[diag_bitmap_3]);

                    rd0 = (__m256d)_mm256_permutevar8x32_epi32((__m256i)rd0, perm_lookup[diag_bitmap_0]);
                    rd1 = (__m256d)_mm256_permutevar8x32_epi32((__m256i)rd1, perm_lookup[diag_bitmap_1]);
                    rd2 = (__m256d)_mm256_permutevar8x32_epi32((__m256i)rd2, perm_lookup[diag_bitmap_2]);
                    rd3 = (__m256d)_mm256_permutevar8x32_epi32((__m256i)rd3, perm_lookup[diag_bitmap_3]);

                    ry0 = _mm256_fmadd_pd(rx0, rd0, ry0);
                    ry1 = _mm256_fmadd_pd(rx1, rd1, ry1);
                    ry2 = _mm256_fmadd_pd(rx2, rd2, ry2);
                    ry3 = _mm256_fmadd_pd(rx3, rd3, ry3);

                    values += pp_4;
                }

                __m256d ry = _mm256_load_pd(y);
                ry0 = _mm256_add_pd(ry0, ry1);
                ry2 = _mm256_add_pd(ry2, ry3);
                ry0 = _mm256_add_pd(ry0, ry2);
                ry = _mm256_add_pd(ry, ry0);
                _mm256_store_pd(y, ry);
                y += block_size;
            }
        }

/*
        void spmv_basic_col(const real_type* x, real_type* y)
        {
            uint64_t* bitmaps = bitmaps_;
            real_type* values = values_;

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    const real_type* x_ptr = x + (bcol << block_size_exp);

                    uint64_t block_bitmap = *bitmaps;
                    bitmaps++;

                    uint8_t col_bitmaps[8] = 
                    for (uint64_t lcol = 0; lcol < block_size; lcol++) {




                    for (uint64_t lrow = 0; lrow < block_size; lrow++) {
                        uint64_t row_bitmap = block_bitmap & 0xFF;
                        block_bitmap >>= 8;

                        if (row_bitmap > 0) {
                            for (uint64_t lcol = 0; lcol < block_size; lcol++) {
                                if (row_bitmap & 0x01) {
                                    *(y + lrow) += *values * *(x_ptr + lcol);
                                    values++;
                                }
                                row_bitmap >>= 1;
                            }
                        }
                    }
                }
                y += block_size;
            }
        }
*/




#if 0
        void spmv(const real_type* RESTRICT x_, real_type* RESTRICT y_)
        {
            const real_type* RESTRICT x = (real_type*)__builtin_assume_aligned(x_, 32);
            real_type* RESTRICT y = (real_type*)__builtin_assume_aligned(y_, 32);
            uint64_t* RESTRICT bitmaps = bitmaps_;
            real_type* RESTRICT values = values_; // (real_type*)__builtin_assume_aligned(values_, 32);

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    uint64_t bitmap = *bitmaps;
                    bitmaps++;

                    const real_type* RESTRICT x_ptr =
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

                y += block_size;
            }
        }

        void spmv_2(const real_type* RESTRICT x_, real_type* RESTRICT y_)
        {
            const real_type* RESTRICT x = (real_type*)__builtin_assume_aligned(x_, 32);
            real_type* RESTRICT y = (real_type*)__builtin_assume_aligned(y_, 32);
            const uint64_t* RESTRICT bitmaps = bitmaps_;
            real_type* RESTRICT values = values_; // (real_type*)__builtin_assume_aligned(values_, 32);

            for (uint64_t brow = 0; brow < num_blocks; brow++) {
                for (uint64_t bcol = 0; bcol < num_blocks; bcol++) {
                    uint64_t bitmap = *bitmaps;
                    bitmaps++;

                    const real_type* RESTRICT x_ptr =
                       (real_type*)__builtin_assume_aligned(x + bcol * block_size, 32);

                    static_assert(block_size == 8, "Block size expected to equal 8.");

                    for (uint64_t qrow = 0; qrow < 2; qrow++) {
                        real_type q_y[4];
                     // memcpy(q_y, y, 4 * sizeof(real_type));
                        for (int i = 0; i < 4; i++)
                            q_y[i] = *(y + i);

                        const real_type* RESTRICT q_x = x_ptr;
                        
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

             // y += block_size;
            }
        }
#endif

    private:
        uint16_t* RESTRICT bitmaps_; // bitmap for each block in lexicographical order
        real_type* RESTRICT values_; // same as for block_coo_matrix

        const uint8_t __attribute__((aligned(64))) popcnt_lookup[16]
            = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

        __m256i __attribute__((aligned(64))) mask_lookup[16];
        __m256i __attribute__((aligned(64))) perm_lookup[16];
};

int main(int argc, char* argv[])
{
    std::cout << "Blocks size: " << magenta << block_size << reset << std::endl;
    std::cout << "Number of block rows/columns: " << magenta << num_blocks << reset << std::endl;
    std::cout << "Number of random experiments: " << magenta << num_experiments << reset << std::endl;
    std::cout << "Number of iterations: " << magenta << num_iterations << reset << std::endl;
    std::cout << "Number of warm-up iterations: " << magenta << warm_up_iterations << reset << std::endl;
    std::cout << "FP precision: " << magenta << (sizeof(real_type) * 8) << reset << std::endl;

    // types of bitmap spmv variant
    enum bitmap_spmv_t {
        bitmap_spmv_basic,
        bitmap_spmv_basic_row,
        bitmap_spmv_basic_row_2,
        bitmap_spmv_popcnt_lzcnt_blsr,
        bitmap_spmv_temp
    } type = bitmap_spmv_basic;

    if (argc > 1)
        type = static_cast<bitmap_spmv_t>(atol(argv[1]));

    std::cout << "Bitmap SpMV type: " << yellow;
    switch (type) {
        case bitmap_spmv_basic:             std::cout << "BITMAP BASIC"; break;
        case bitmap_spmv_basic_row:         std::cout << "BITMAP BASIC ROW"; break;
        case bitmap_spmv_basic_row_2:       std::cout << "BITMAP BASIC ROW 2"; break;
        case bitmap_spmv_popcnt_lzcnt_blsr: std::cout << "BITMAP POPCNT_LZCNT_BLSR"; break;
        case bitmap_spmv_temp:              std::cout << "BITMAP TEMP"; break;
        default: throw std::runtime_error("Unknown matrix/spmv type!");
    }
    std::cout << reset << std::endl;

    block_generator bgen;

    uint64_t const n = num_blocks * block_size;
 // real_type* x_ = (real_type*)aligned_alloc(64, n * sizeof(real_type));
 // real_type* y_ = (real_type*)aligned_alloc(64, n * sizeof(real_type));
    real_type *x, *y;
    posix_memalign((void**)(&x), 64, n * sizeof(real_type));
    posix_memalign((void**)(&y), 64, n * sizeof(real_type));
    for (uint64_t k = 0; k < n; k++) {
        x[k] = 1.0;
        y[k] = 0.0;
    }

    block_bitmap_matrix_t block_bitmap_matrix;

    for (int bnnz = 1; bnnz <= block_size * block_size; bnnz++) {
        double mflops_min, mflops_max, mflops_avg, mflops_sum = 0.0;

        for (int exp = 0; exp < num_experiments; exp++) {
            bgen.generate_random(bnnz);
            block_bitmap_matrix.assemble(bgen);

            timer_type timer;

            for (int iter = -warm_up_iterations; iter < num_iterations; iter++) {
                if (iter == 0)
                    timer.start();

                switch (type) {
                    case bitmap_spmv_basic:             block_bitmap_matrix.spmv_basic(x, y); break;
                    case bitmap_spmv_basic_row:         block_bitmap_matrix.spmv_basic_row(x, y); break;
                    case bitmap_spmv_basic_row_2:       block_bitmap_matrix.spmv_basic_row_2(x, y); break;
                    case bitmap_spmv_popcnt_lzcnt_blsr: block_bitmap_matrix.spmv_popcnt_lzcnt_blsr(x, y); break;
                    case bitmap_spmv_temp:              block_bitmap_matrix.spmv_temp(x, y); break;
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
