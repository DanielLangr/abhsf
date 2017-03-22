#ifndef UTILS_SYNTHETIC_H
#define UTILS_SYNTHETIC_H

#include <random>

#include "matrix_properties.h"

template <typename ELEMENTS_T>
void synthetise_dense_matrix(matrix_properties& props, ELEMENTS_T& elements, long n)
{
    props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
    props.type = matrix_type_t::REAL;
    props.m = props.n = n;
    props.nnz = props.m * props.n;

    elements.reserve(props.nnz);
    for (long i = 0; i < props.m; i++)
        for (long j = 0; j < props.n; j++)
            elements.emplace_back(i, j, 1.0);
}

template <typename ELEMENTS_T>
void synthetise_diagonal_matrix(matrix_properties& props, ELEMENTS_T& elements, long n)
{
    props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
    props.type = matrix_type_t::REAL;
    props.m = props.n = n;
    props.nnz = props.m;

    elements.reserve(props.nnz);
    for (size_t i = 0; i < props.m; i++)
        elements.emplace_back(i, i, 1.0);
}

template <typename ELEMENTS_T>
void synthetise_single_column_matrix(matrix_properties& props, ELEMENTS_T& elements, long n)
{
    props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
    props.type = matrix_type_t::REAL;
    props.m = props.n = n;
    props.nnz = props.m;

    elements.reserve(props.nnz);
    for (size_t i = 0; i < props.m; i++)
        elements.emplace_back(i, 0, 1.0);
}

template <typename ELEMENTS_T>
void synthetise_random_column_matrix(matrix_properties& props, ELEMENTS_T& elements, long n)
{
    props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
    props.type = matrix_type_t::REAL;
    props.m = props.n = n;
    props.nnz = props.m;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, props.n - 1);

    elements.reserve(props.nnz);
    for (size_t i = 0; i < props.m; i++)
        elements.emplace_back(i, dist(mt), 1.0);
}

template <typename ELEMENTS_T>
void synthetise_single_row_column_matrix(matrix_properties& props, ELEMENTS_T& elements, long n)
{
    props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
    props.type = matrix_type_t::REAL;
    props.m = props.n = n;
    props.nnz = props.m + props.n - 1;

    elements.reserve(props.nnz);
    for (size_t j = 0; j < props.n; j++)
        elements.emplace_back(0, j, 1.0);
    for (size_t i = 1; i < props.m; i++)
        elements.emplace_back(i, 0, 1.0);
}

template <typename ELEMENTS_T>
void synthetise_single_row_matrix(matrix_properties& props, ELEMENTS_T& elements, long n)
{
    props.symmetry = matrix_symmetry_t::UNSYMMETRIC;
    props.type = matrix_type_t::REAL;
    props.m = props.n = n;
    props.nnz = props.n;

    elements.reserve(props.nnz);
    for (size_t i = 0; i < props.n; i++)
        elements.emplace_back(props.m - 1, i, 1.0);
}

#endif
