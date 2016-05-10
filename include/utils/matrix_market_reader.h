#ifndef UTILS_MATRIX_MARKET_READER_H
#define UTILS_MATRIX_MARKET_READER_H

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace 
{
#include "mmio_modified/mmio.c"
}

#include "colors.h"
#include "matrix_properties.h"

template <typename T = std::ostream>
class matrix_market_reader 
{
    public:
        explicit matrix_market_reader(T* log = nullptr) : f_(nullptr), log_(log) { }
        ~matrix_market_reader() { close(); }

        void open(const std::string& filename);
        void next_element(uintmax_t* row = nullptr, uintmax_t* col = nullptr,
                double* val_re = nullptr, double* val_im = nullptr, intmax_t* val_int = nullptr);
        void close();

        const matrix_properties& props() const { return props_; }

    private:
        FILE* f_;
        T* log_;

        matrix_properties props_;
};

template <typename T>
void matrix_market_reader<T>::open(const std::string& filename)
{
    if ((f_ = fopen(filename.c_str(), "r")) == 0) 
        throw std::runtime_error(std::string("Cannot open input file: ") + filename);
    if (log_) *log_ << "File " << yellow << filename << reset << " successfully opened..." << std::endl;

    MM_typecode matcode;

    if (mm_read_banner(f_, &matcode) != 0) 
        throw std::runtime_error("Could not process Matrix Market banner.");

    if ((mm_is_matrix(matcode) == 0) || (mm_is_coordinate(matcode) == 0)) 
        throw std::runtime_error("Only sparse matrices in coordinate format are supported.");

    if (mm_is_pattern(matcode)) 
        props_.type = matrix_type_t::BINARY;
    else if (mm_is_integer(matcode)) 
        props_.type = matrix_type_t::INTEGER;
    else if (mm_is_real(matcode)) 
        props_.type = matrix_type_t::REAL;
    else if (mm_is_complex(matcode)) 
        props_.type = matrix_type_t::COMPLEX;
    else
        throw std::runtime_error("Unsupported matrix type.");

    if (mm_is_symmetric(matcode)) 
        props_.symmetry = matrix_symmetry_t::SYMMETRIC;
    else if (mm_is_hermitian(matcode)) 
        props_.symmetry = matrix_symmetry_t::HERMITIAN;
    else if (mm_is_skew(matcode)) 
        props_.symmetry = matrix_symmetry_t::SKEW_SYMMETRIC;
    else if (mm_is_general(matcode)) 
        props_.symmetry = matrix_symmetry_t::UNSYMMETRIC;
    else 
        throw std::runtime_error("Unsupported type of matrix symmetry.");

    if (log_) *log_ << "Type of matrix: " << cyan << props_.shape_str() << ", "
        << props_.type_str() << ", " << props_.symmetry_str() << reset << std::endl;
    int m, n, nnz;
    if (mm_read_mtx_crd_size(f_, &m, &n, &nnz) != 0) 
        throw std::runtime_error("Could not retrieve matrix sizes.");

    props_.m = m;
    props_.n = n;
    props_.nnz = nnz;

    if (log_) *log_ << "Number of rows:             "
        << cyan << std::right << std::setw(20) << props_.m << reset << std::endl;
    if (log_) *log_ << "Number of columns:          "
        << cyan << std::right << std::setw(20) << props_.n << reset << std::endl;
    if (log_) *log_ << "Number of nonzero elements: "
        << cyan << std::right << std::setw(20) << props_.nnz << reset << std::endl;
}

template <typename T>
void matrix_market_reader<T>::next_element(uintmax_t* row, uintmax_t* col,
        double* val_re, double* val_im, intmax_t* val_int)
{
    unsigned long row_, col_;
    double val_re_ = 0.0, val_im_ = 0.0;
    intmax_t val_int_ = 0;

    switch (props_.type) {
        case matrix_type_t::BINARY:
            if (fscanf(f_, "%lu %lu", &row_, &col_) != 2) 
                throw std::runtime_error("Could not read matrix element.");
            break;

        case matrix_type_t::INTEGER:
            if (fscanf(f_, "%lu %lu %ld", &row_, &col_, &val_int_) != 3) 
                throw std::runtime_error("Could not read matrix element.");
            break;

        case matrix_type_t::REAL:
            if (fscanf(f_, "%lu %lu %lf", &row_, &col_, &val_re_) != 3) 
                throw std::runtime_error("Could not read matrix element.");
            break;

        case matrix_type_t::COMPLEX:
            if (fscanf(f_, "%lu %lu %lf %lf", &row_, &col_, &val_re_, &val_im_) != 4) 
                throw std::runtime_error("Could not read matrix element.");
            break;

        default:
            assert(false);
    }

    if (val_re) *val_re = val_re_;
    if (val_im) *val_im = val_im_;
    if (val_int) *val_int = val_int_;
    if (row) *row = uintmax_t(row_ - 1);
    if (col) *col = uintmax_t(col_ - 1);
}

template <typename T>
void matrix_market_reader<T>::close()
{
    if (f_) {
        fclose(f_);
        f_ = nullptr;
    }
}

#endif
