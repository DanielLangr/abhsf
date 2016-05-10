#ifndef UTILS_MATRIX_PROPERTIES_H
#define UTILS_MATRIX_PROPERTIES_H

#include <cassert>
#include <cstdint>
#include <string>

enum class matrix_type_t
{
    BINARY,
    INTEGER,
    REAL,
    COMPLEX
};

enum class matrix_symmetry_t
{
    SYMMETRIC,
    HERMITIAN,
    SKEW_SYMMETRIC,
    UNSYMMETRIC
};

struct matrix_properties
{
    matrix_type_t type;
    matrix_symmetry_t symmetry;

    uintmax_t m, n, nnz;

    std::string type_str()
    {
        switch(type) {
            case matrix_type_t::BINARY:
                return std::string("binary");
            case matrix_type_t::INTEGER:
                return std::string("integer");
            case matrix_type_t::REAL:
                return std::string("real");
            case matrix_type_t::COMPLEX:
                return std::string("complex");
            default:
                assert(false);
        }

        return std::string("unknown");
    }

    std::string symmetry_str() 
    {
        switch (symmetry) {
            case matrix_symmetry_t::SYMMETRIC:
                return std::string("symmetric");
            case matrix_symmetry_t::HERMITIAN:
                return std::string("Hermitian");
            case matrix_symmetry_t::SKEW_SYMMETRIC:
                return std::string("skew-symmetric");
            case matrix_symmetry_t::UNSYMMETRIC:
                return std::string("unsymmetric");
            default:
                assert(false);
        }

        return std::string("unknown");
    }

    std::string shape_str()
    {
        return (m == n) ? std::string("square") : std::string("rectangular");
    }
};

#endif
