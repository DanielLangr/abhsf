#ifndef UTILS_RESTRICT_H
#define UTILS_RESTRICT_H

#ifdef __INTEL_COMPILER
    #define RESTRICT restrict
#elif defined __GNUC__
    #define RESTRICT __restrict__
#else
    #define RESTRICT 
#endif

#endif
