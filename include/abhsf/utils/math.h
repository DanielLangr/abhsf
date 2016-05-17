#ifndef ABHSF_UTILS_MATH_H
#define ABHSF_UTILS_MATH_H

#include <cassert>

template <typename T>
T ceil_div(T divident, T divisor)
{
    assert(divisor > 0);

    if (divident % divisor)
        return divident / divisor + 1;
    else 
        return divident / divisor;
} 

template <typename T>
T ceil_log2(T n)
{
    assert(n > 0);

    T k = 0;
    while ((1UL << k) < n)
        k++;
    
    assert((1UL << k) >= n);
    assert((1UL << (k - 1)) < n);

    return k;
}

#endif
