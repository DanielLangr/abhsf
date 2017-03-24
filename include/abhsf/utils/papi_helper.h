#ifndef UTILS_PAPI_HELPER_H
#define UTILS_PAPI_HELPER_H

#include <omp.h>

#include <iostream>
#include <stdexcept>

#include <pthread.h>

#ifdef HAVE_PAPI
    #include <papi.h>
#endif

#include "colors.h"

static long long l1_tcm, l2_tcm, l3_tcm, tlb_dm;

void papi_helper_init_thread()
{
#ifdef HAVE_PAPI
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        throw std::runtime_error("Error initializing PAPI library!");
    if (PAPI_thread_init(pthread_self) != PAPI_OK)
        throw std::runtime_error("Error setting threading support for PAPI!");
#endif
}

void papi_helper_start()
{
#ifdef HAVE_PAPI
#pragma omp single
    l1_tcm = l2_tcm = l3_tcm = tlb_dm = 0;

    int papi_events[4] = { PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM, PAPI_TLB_DM };
    if (PAPI_start_counters(papi_events, 4) != PAPI_OK)
        throw std::runtime_error("Error running PAPI_start_counters function!");
#endif
}

void papi_helper_stop()
{
#ifdef HAVE_PAPI
    long long papi_values[4];
    if (PAPI_stop_counters(papi_values, 4) != PAPI_OK)
        throw std::runtime_error("Error running PAPI_stop_counters function!");

#pragma omp atomic update
    l1_tcm += papi_values[0];
#pragma omp atomic update
    l2_tcm += papi_values[1];
#pragma omp atomic update
    l3_tcm += papi_values[2];
#pragma omp atomic update
    tlb_dm += papi_values[3];
#endif
}

void papi_helper_print()
{
#ifdef HAVE_PAPI
    std::cout << "L1 cache misses: "
        << cyan << std::right << std::setw(20) << l1_tcm << reset << std::endl;
    std::cout << "L2 cache misses: "
        << cyan << std::right << std::setw(20) << l2_tcm << reset << std::endl;
    std::cout << "L3 cache misses: "
        << cyan << std::right << std::setw(20) << l3_tcm << reset << std::endl;
    std::cout << "TLB misses:      "
        << cyan << std::right << std::setw(20) << tlb_dm << reset << std::endl;
#endif
}

#endif
