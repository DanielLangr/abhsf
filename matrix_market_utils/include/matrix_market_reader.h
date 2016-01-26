#ifndef MATRIX_MARKET_READER_H
#define MATRIX_MARKET_READER_H

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace 
{
#include "mmio_modified/mmio.c"
}

template <typename T = std::ostream>
class matrix_market_reader 
{
    public:
        explicit matrix_market_reader(T* log = nullptr) : f_(nullptr), log_(log) { }
        ~matrix_market_reader() { close(); }

        void open(const std::string& filename);
        void next_element(uintmax_t* row, uintmax_t* col, double* re, double* im = 0);
        void close();

        bool c; // is complex ?
        bool h; // is Hermitian ?
        enum { PATTERN, INTEGER, REAL, COMPLEX } type;

        uintmax_t m; // number of rows
        uintmax_t n; // number of columns
        uintmax_t z; // number of nonzeros

    private:
        FILE* f_;
        T* log_;
};

template <typename T>
void matrix_market_reader<T>::open(const std::string& filename)
{
    if ((f_ = fopen(filename.c_str(), "r")) == 0) 
        throw std::runtime_error(std::string("Cannot open input file: ") + filename);
    if (log_) *log_ << "File " << filename << " successfully opened..." << std::endl;

    MM_typecode matcode;

    if (mm_read_banner(f_, &matcode) != 0) 
        throw std::runtime_error("Could not process Matrix Market banner.");

    if ((mm_is_matrix(matcode) == 0) || (mm_is_coordinate(matcode) == 0)) 
        throw std::runtime_error("Only sparse matrices in coordinate format are supported.");

    if (mm_is_real(matcode)) {
        c = false;
        type = REAL;

        if (mm_is_symmetric(matcode)) {
            h = true;
            if (log_) *log_ << "Type of matrix: REAL, SYMMETRIC";
        }
        else if (mm_is_general(matcode)) {
            h = false;
            if (log_) *log_ << "Type of matrix: REAL, UNSYMMETRIC";
        }
        else 
            throw std::runtime_error("Only real symmetric or unsymmetric matrices are supported.");
    }
    else if (mm_is_complex(matcode)) {
        c = true;
        type = COMPLEX;

        if (mm_is_hermitian(matcode)) {
            h = true;
            if (log_) *log_ << "Type of matrix: COMPLEX, HERMITIAN";
        }
        else if (mm_is_general(matcode)) {
            h = false;
            if (log_) *log_ << "Type of matrix: COMPLEX, UNSYMMETRIC";
        }
        else 
            throw std::runtime_error("Only complex Hermitian or unsymmetric matrices are supported.");
    }
    else if (mm_is_integer(matcode)) {
        c = false; 
        type = INTEGER;

        if (mm_is_symmetric(matcode)) {
            h = true;
            if (log_) *log_ << "Type of matrix: INTEGER, SYMMETRIC";
        }
        else if (mm_is_general(matcode)) {
            h = false;
            if (log_) *log_ << "Type of matrix: INTEGER, UNSYMMETRIC";
        }
        else 
            throw std::runtime_error("Only real symmetric or unsymmetric matrices are supported.");
    }
    else if (mm_is_pattern(matcode)) {
        c = false; 
        type = PATTERN;

        if (mm_is_symmetric(matcode)) {
            h = true;
            if (log_) *log_ << "Type of matrix: PATTERN, SYMMETRIC";
        }
        else if (mm_is_general(matcode)) {
            h = false;
            if (log_) *log_ << "Type of matrix: PATTERN, UNSYMMETRIC";
        }
        else 
            throw std::runtime_error("Only real symmetric or unsymmetric matrices are supported.");
    }
    else
        throw std::runtime_error("Only real or complex sparse matrices are supported.");

    int m, n, z;

    if (mm_read_mtx_crd_size(f_, &m, &n, &z) != 0) 
        throw std::runtime_error("Could not process matrix sizes.");

    if (log_) {
        if (m == n)
            *log_ << ", SQUARE" << std::endl;
        else
            *log_ << ", RECTANGULAR" << std::endl;
    }

    this->m = m;
    this->n = n;
    this->z = z;

    if (log_) *log_ << "Number of rows:             " << std::right << std::setw(20) << m << std::endl;
    if (log_) *log_ << "Number of columns:          " << std::right << std::setw(20) << n << std::endl;
    if (log_) *log_ << "Number of nonzero elements: " << std::right << std::setw(20) << z << std::endl;
}

template <typename T>
void matrix_market_reader<T>::next_element(uintmax_t* row, uintmax_t* col, double* re, double* im)
{
    unsigned long row_, col_;
    double re_, im_ = 0.0;
    long int_;
  
    if ((c) && (type == COMPLEX)) {
        if (fscanf(f_, "%lu %lu %lf %lf", &row_, &col_, &re_, &im_) != 4) 
            throw std::runtime_error("Could not read matrix element.");
        if (re) *re = re_;
        if (im) *im = im_;
    }
    else if ((!c) && (type == REAL)) {
        if (fscanf(f_, "%lu %lu %lf", &row_, &col_, &re_) != 3) 
            throw std::runtime_error("Could not read matrix element.");
        if (re) *re = re_;
    }
    else if ((!c) && (type == INTEGER)) {
        if (fscanf(f_, "%lu %lu %ld", &row_, &col_, &int_) != 3) 
            throw std::runtime_error("Could not read matrix element.");
        if (re) *re = static_cast<double>(int_);
    }
    else if ((!c) && (type == PATTERN)) {
        if (fscanf(f_, "%lu %lu", &row_, &col_) != 2) 
            throw std::runtime_error("Could not read matrix element.");
        if (re) *re = 1.0;
    }
    else
        throw std::runtime_error("Unsupported matrix type.");

    *row = static_cast<uintmax_t>(row_);
    *col = static_cast<uintmax_t>(col_);
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
