#pragma once

#include <cassert>
#include <random>
#include <tuple>
#include <vector>

#include <parallel/algorithm>

template <typename index_type = int, typename real_type = double>
class csr_synth
{
    public:
        void create_dense(long n) 
        {
            m_ = n_= n;
            nnz_ = m_ * n_;

            elements_t elements;
            elements.reserve(nnz_);
            for (size_t i = 0; i < m_; i++)
                for (size_t j = 0; j < n_; j++)
                    elements.emplace_back(i, j, 1.0);

            from_elements(elements);
        }

        void create_diagonal(long n)
        {
            m_ = n_= n;
            nnz_ = m_;

            elements_t elements;
            elements.reserve(nnz_);
            for (size_t i = 0; i < m_; i++)
                elements.emplace_back(i, i, 1.0);

            from_elements(elements);
        }

        void create_single_column(long n)
        {
            m_ = n_= n;
            nnz_ = m_;

            elements_t elements;
            elements.reserve(nnz_);
            for (size_t i = 0; i < m_; i++)
                elements.emplace_back(i, 0, 1.0);

            from_elements(elements);
        }

        void create_random_column(long n)
        {
            m_ = n_= n;
            nnz_ = m_;

            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_int_distribution<int> dist(0, n_ - 1);

            elements_t elements;
            elements.reserve(nnz_);
            for (size_t i = 0; i < m_; i++)
                elements.emplace_back(i, dist(mt), 1.0);

            from_elements(elements);
        }

        void create_single_row_column(long n)
        {
            m_ = n_= n;
            nnz_ = m_ + n_ - 1;

            elements_t elements;
            elements.reserve(nnz_);
            for (size_t j = 0; j < n_; j++)
                elements.emplace_back(0, j, 1.0);
            for (size_t i = 1; i < m_; i++)
                elements.emplace_back(i, 0, 1.0);

            from_elements(elements);
        }

        void create_single_row(long n)
        {
            m_ = n_= n;
            nnz_ = n_;

            elements_t elements;
            elements.reserve(nnz_);
            for (size_t i = 0; i < n_; i++)
                elements.emplace_back(m_ - 1, i, 1.0);
        
            from_elements(elements);
        }

        index_type m() const { return m_; }
        index_type n() const { return n_; }
        index_type nnz() const { return nnz_; }

        const index_type* row_ptrs() const { return ia_.data(); }
        const index_type* col_inds() const { return ja_.data(); }
        const real_type* vals() const { return a_.data(); }

    private:
        using element_t = std::tuple<index_type, index_type, real_type>;
        using elements_t = std::vector<element_t>;

        void from_elements(elements_t& elements)
        {
         // std::sort(elements.begin(), elements.end());
            __gnu_parallel::sort(elements.begin(), elements.end());

            a_.resize(nnz_);
            ia_.resize(m_ + 1);
            ja_.resize(nnz_);

            for (size_t k = 0; k < nnz_; k++) {
                a_[k] = std::get<2>(elements[k]);
                ja_[k] = std::get<1>(elements[k]);
            }

            ia_[0] = 0;
            size_t k = 0;
            size_t row = 0;

            while (k < nnz_) {
                while ((k < nnz_) && (row == std::get<0>(elements[k])))
                    k++;

                row++;
                ia_[row] = k;
            }

            assert(ia_[m_] == nnz_);
        }

        std::vector<real_type> a_;
        std::vector<index_type> ia_;
        std::vector<index_type> ja_;
        index_type m_, n_, nnz_;
};
