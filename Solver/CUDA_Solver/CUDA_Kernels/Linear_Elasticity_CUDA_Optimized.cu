//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#include <iostream>
#include <thrust/device_vector.h>
#include "Linear_Elasticity_CUDA_Optimized.h"
using namespace SPGrid;
#define THREADBLOCK 512

__constant__ float K_mu_device[576];
__constant__ float K_la_device[576];


//template <typename T,typename T_offset_ptr>
struct Parameters{
    float* f[3];
    const float* u[3];
    const float* mu;
    const float* lambda;
    const unsigned int* b;
    unsigned int number_of_blocks;
};

// TODO: Remove the explicit template parameters on this one.
__constant__ char p_device[sizeof(Parameters)];

bool symbol_initialized = false;

// Index here: f_i^v = Sum_{jw}(K[i][j][v][w])
const float  __attribute__ ((aligned(32))) K_mu[8][8][3][3] =
{
    {
        {
            {      32.f/72.f,        6.f/72.f,        6.f/72.f},
            {       6.f/72.f,       32.f/72.f,        6.f/72.f},
            {       6.f/72.f,        6.f/72.f,       32.f/72.f}
        },
        {
            {       4.f/72.f,        3.f/72.f,        6.f/72.f},
            {       3.f/72.f,        4.f/72.f,        6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        },
        {
            {       4.f/72.f,        6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        6.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,      -10.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,      -10.f/72.f}
        },
        {
            {      -8.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        4.f/72.f,        3.f/72.f},
            {       6.f/72.f,        3.f/72.f,        4.f/72.f}
        },
        {
            {     -10.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,       -4.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,      -10.f/72.f}
        },
        {
            {     -10.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,      -10.f/72.f,       -3.f/72.f},
            {       3.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -8.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,       -8.f/72.f}
        }
    },
    {
        {
            {       4.f/72.f,        3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        6.f/72.f,       -8.f/72.f}
        },
        {
            {      32.f/72.f,        6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,       32.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,       32.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,      -10.f/72.f,        6.f/72.f},
            {       3.f/72.f,        6.f/72.f,      -10.f/72.f}
        },
        {
            {       4.f/72.f,        6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,        4.f/72.f}
        },
        {
            {     -10.f/72.f,       -3.f/72.f,        6.f/72.f},
            {       3.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {       6.f/72.f,        3.f/72.f,      -10.f/72.f}
        },
        {
            {      -8.f/72.f,       -6.f/72.f,        6.f/72.f},
            {       6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {      -8.f/72.f,       -3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -8.f/72.f,        3.f/72.f},
            {       3.f/72.f,        3.f/72.f,       -8.f/72.f}
        },
        {
            {     -10.f/72.f,       -6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,      -10.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,       -4.f/72.f}
        }
    },
    {
        {
            {       4.f/72.f,       -6.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,       -3.f/72.f,        3.f/72.f},
            {       3.f/72.f,      -10.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,      -10.f/72.f}
        },
        {
            {      32.f/72.f,       -6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,       32.f/72.f,       -6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,       32.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,       -8.f/72.f}
        },
        {
            {     -10.f/72.f,        6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,      -10.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,        3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -8.f/72.f,        3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,       -8.f/72.f}
        },
        {
            {      -8.f/72.f,        6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {     -10.f/72.f,        3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,      -10.f/72.f}
        }
    },
    {
        {
            {      -4.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,      -10.f/72.f,       -6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,      -10.f/72.f}
        },
        {
            {       4.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,        4.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,        6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        },
        {
            {      32.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,       32.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,       32.f/72.f}
        },
        {
            {      -8.f/72.f,        3.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -8.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,       -8.f/72.f}
        },
        {
            {     -10.f/72.f,        6.f/72.f,        3.f/72.f},
            {       6.f/72.f,      -10.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {     -10.f/72.f,        3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,      -10.f/72.f}
        },
        {
            {      -8.f/72.f,        6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,        3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,        4.f/72.f}
        }
    },
    {
        {
            {      -8.f/72.f,        6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,        3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,        4.f/72.f}
        },
        {
            {     -10.f/72.f,        3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,      -10.f/72.f}
        },
        {
            {     -10.f/72.f,        6.f/72.f,        3.f/72.f},
            {       6.f/72.f,      -10.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,        3.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -8.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,       -8.f/72.f}
        },
        {
            {      32.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,       32.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,       32.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,        6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        },
        {
            {       4.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,      -10.f/72.f,       -6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,      -10.f/72.f}
        }
    },
    {
        {
            {     -10.f/72.f,        3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,      -10.f/72.f}
        },
        {
            {      -8.f/72.f,        6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {      -8.f/72.f,        3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -8.f/72.f,        3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,       -8.f/72.f}
        },
        {
            {     -10.f/72.f,        6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,      -10.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,       -4.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,       -8.f/72.f}
        },
        {
            {      32.f/72.f,       -6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,       32.f/72.f,       -6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,       32.f/72.f}
        },
        {
            {      -4.f/72.f,       -3.f/72.f,        3.f/72.f},
            {       3.f/72.f,      -10.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,      -10.f/72.f}
        },
        {
            {       4.f/72.f,       -6.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,        4.f/72.f}
        }
    },
    {
        {
            {     -10.f/72.f,       -6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,      -10.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,       -3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -8.f/72.f,        3.f/72.f},
            {       3.f/72.f,        3.f/72.f,       -8.f/72.f}
        },
        {
            {      -8.f/72.f,       -6.f/72.f,        6.f/72.f},
            {       6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {     -10.f/72.f,       -3.f/72.f,        6.f/72.f},
            {       3.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {       6.f/72.f,        3.f/72.f,      -10.f/72.f}
        },
        {
            {       4.f/72.f,        6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,      -10.f/72.f,        6.f/72.f},
            {       3.f/72.f,        6.f/72.f,      -10.f/72.f}
        },
        {
            {      32.f/72.f,        6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,       32.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,       32.f/72.f}
        },
        {
            {       4.f/72.f,        3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        6.f/72.f,       -8.f/72.f}
        }
    },
    {
        {
            {      -8.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -8.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,       -8.f/72.f}
        },
        {
            {     -10.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,      -10.f/72.f,       -3.f/72.f},
            {       3.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {     -10.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,       -4.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,      -10.f/72.f}
        },
        {
            {      -8.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        4.f/72.f,        3.f/72.f},
            {       6.f/72.f,        3.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,      -10.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,      -10.f/72.f}
        },
        {
            {       4.f/72.f,        6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        6.f/72.f,        4.f/72.f}
        },
        {
            {       4.f/72.f,        3.f/72.f,        6.f/72.f},
            {       3.f/72.f,        4.f/72.f,        6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        },
        {
            {      32.f/72.f,        6.f/72.f,        6.f/72.f},
            {       6.f/72.f,       32.f/72.f,        6.f/72.f},
            {       6.f/72.f,        6.f/72.f,       32.f/72.f}
        }
    }
};
const float  __attribute__ ((aligned(32))) K_lambda[8][8][3][3] =
{
    {
        {
            {       8.f/72.f,        6.f/72.f,        6.f/72.f},
            {       6.f/72.f,        8.f/72.f,        6.f/72.f},
            {       6.f/72.f,        6.f/72.f,        8.f/72.f}
        },
        {
            {       4.f/72.f,        3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        6.f/72.f,       -8.f/72.f}
        },
        {
            {       4.f/72.f,       -6.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,        4.f/72.f}
        },
        {
            {       2.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -4.f/72.f,       -6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,        6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,        3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        2.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {      -4.f/72.f,       -6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -4.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,        2.f/72.f}
        },
        {
            {      -2.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -2.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,       -2.f/72.f}
        }
    },
    {
        {
            {       4.f/72.f,        3.f/72.f,        6.f/72.f},
            {       3.f/72.f,        4.f/72.f,        6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        },
        {
            {       8.f/72.f,        6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        8.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,        8.f/72.f}
        },
        {
            {       2.f/72.f,       -3.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -4.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,       -4.f/72.f}
        },
        {
            {       4.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        2.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,        6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {      -2.f/72.f,       -3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -2.f/72.f,        3.f/72.f},
            {       3.f/72.f,        3.f/72.f,       -2.f/72.f}
        },
        {
            {      -4.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {       3.f/72.f,        3.f/72.f,        2.f/72.f}
        }
    },
    {
        {
            {       4.f/72.f,        6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        6.f/72.f,        4.f/72.f}
        },
        {
            {       2.f/72.f,        3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,        6.f/72.f},
            {       3.f/72.f,        6.f/72.f,       -4.f/72.f}
        },
        {
            {       8.f/72.f,       -6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        8.f/72.f,       -6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,        8.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,        6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        },
        {
            {      -4.f/72.f,        6.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,        2.f/72.f}
        },
        {
            {      -2.f/72.f,        3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -2.f/72.f,        3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,       -2.f/72.f}
        },
        {
            {      -8.f/72.f,       -6.f/72.f,        6.f/72.f},
            {       6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        2.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,       -4.f/72.f}
        }
    },
    {
        {
            {       2.f/72.f,        3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,       -4.f/72.f}
        },
        {
            {       4.f/72.f,        6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,        4.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,       -8.f/72.f}
        },
        {
            {       8.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        8.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,        8.f/72.f}
        },
        {
            {      -2.f/72.f,        3.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -2.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,       -2.f/72.f}
        },
        {
            {      -4.f/72.f,        6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -4.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,        2.f/72.f}
        },
        {
            {      -4.f/72.f,       -3.f/72.f,        6.f/72.f},
            {       3.f/72.f,        2.f/72.f,       -3.f/72.f},
            {       6.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        4.f/72.f,        3.f/72.f},
            {       6.f/72.f,        3.f/72.f,        4.f/72.f}
        }
    },
    {
        {
            {      -8.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        4.f/72.f,        3.f/72.f},
            {       6.f/72.f,        3.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,       -3.f/72.f,        6.f/72.f},
            {       3.f/72.f,        2.f/72.f,       -3.f/72.f},
            {       6.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {      -4.f/72.f,        6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -4.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,        2.f/72.f}
        },
        {
            {      -2.f/72.f,        3.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -2.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -3.f/72.f,       -2.f/72.f}
        },
        {
            {       8.f/72.f,       -6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        8.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,        8.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        6.f/72.f,       -8.f/72.f}
        },
        {
            {       4.f/72.f,        6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,        4.f/72.f}
        },
        {
            {       2.f/72.f,        3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,       -6.f/72.f,       -4.f/72.f}
        }
    },
    {
        {
            {      -4.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        2.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,       -6.f/72.f,        6.f/72.f},
            {       6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {      -2.f/72.f,        3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -2.f/72.f,        3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,       -2.f/72.f}
        },
        {
            {      -4.f/72.f,        6.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,        3.f/72.f,        2.f/72.f}
        },
        {
            {       4.f/72.f,       -3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        4.f/72.f,        6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        },
        {
            {       8.f/72.f,       -6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        8.f/72.f,       -6.f/72.f},
            {       6.f/72.f,       -6.f/72.f,        8.f/72.f}
        },
        {
            {       2.f/72.f,        3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -4.f/72.f,        6.f/72.f},
            {       3.f/72.f,        6.f/72.f,       -4.f/72.f}
        },
        {
            {       4.f/72.f,        6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        6.f/72.f,        4.f/72.f}
        }
    },
    {
        {
            {      -4.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,       -4.f/72.f,       -3.f/72.f},
            {       3.f/72.f,        3.f/72.f,        2.f/72.f}
        },
        {
            {      -2.f/72.f,       -3.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -2.f/72.f,        3.f/72.f},
            {       3.f/72.f,        3.f/72.f,       -2.f/72.f}
        },
        {
            {      -8.f/72.f,        6.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,        4.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        2.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -3.f/72.f,       -4.f/72.f}
        },
        {
            {       4.f/72.f,       -6.f/72.f,       -3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,        4.f/72.f}
        },
        {
            {       2.f/72.f,       -3.f/72.f,        3.f/72.f},
            {       3.f/72.f,       -4.f/72.f,        6.f/72.f},
            {      -3.f/72.f,        6.f/72.f,       -4.f/72.f}
        },
        {
            {       8.f/72.f,        6.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        8.f/72.f,       -6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,        8.f/72.f}
        },
        {
            {       4.f/72.f,        3.f/72.f,        6.f/72.f},
            {       3.f/72.f,        4.f/72.f,        6.f/72.f},
            {      -6.f/72.f,       -6.f/72.f,       -8.f/72.f}
        }
    },
    {
        {
            {      -2.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -2.f/72.f,       -3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,       -2.f/72.f}
        },
        {
            {      -4.f/72.f,       -6.f/72.f,        3.f/72.f},
            {      -6.f/72.f,       -4.f/72.f,        3.f/72.f},
            {      -3.f/72.f,       -3.f/72.f,        2.f/72.f}
        },
        {
            {      -4.f/72.f,        3.f/72.f,       -6.f/72.f},
            {      -3.f/72.f,        2.f/72.f,       -3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,       -4.f/72.f}
        },
        {
            {      -8.f/72.f,        6.f/72.f,        6.f/72.f},
            {      -6.f/72.f,        4.f/72.f,        3.f/72.f},
            {      -6.f/72.f,        3.f/72.f,        4.f/72.f}
        },
        {
            {       2.f/72.f,       -3.f/72.f,       -3.f/72.f},
            {       3.f/72.f,       -4.f/72.f,       -6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,       -4.f/72.f}
        },
        {
            {       4.f/72.f,       -6.f/72.f,        3.f/72.f},
            {       6.f/72.f,       -8.f/72.f,        6.f/72.f},
            {       3.f/72.f,       -6.f/72.f,        4.f/72.f}
        },
        {
            {       4.f/72.f,        3.f/72.f,       -6.f/72.f},
            {       3.f/72.f,        4.f/72.f,       -6.f/72.f},
            {       6.f/72.f,        6.f/72.f,       -8.f/72.f}
        },
        {
            {       8.f/72.f,        6.f/72.f,        6.f/72.f},
            {       6.f/72.f,        8.f/72.f,        6.f/72.f},
            {       6.f/72.f,        6.f/72.f,        8.f/72.f}
        }
    }
};
template<class T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__device__ T Lookup_Value_3D(const T* a,const T_offset_ptr b[27],int x,int y,int z)
{
    int block_id = 13;
    if(z < 0) {block_id -= 1;z += block_zsize;}
    if(z >= block_zsize) {block_id += 1;z -= block_zsize;}
    if(y < 0) {block_id -= 3;y += block_ysize;}
    if(y >= block_ysize) {block_id += 3;y -= block_ysize;}
    if(x < 0) {block_id -= 9;x += block_xsize;}
    if(x >= block_xsize) {block_id += 9;x -= block_xsize;}
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    const int entry = z + y * ystride + x * xstride;
    return reinterpret_cast<T*>((unsigned long)a + (unsigned long)b[block_id])[entry];
}

template<class T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__device__ void Lookup_Two_Values_3D(T& a_out,T& b_out, const T* a_array, const T* b_array, const T_offset_ptr b[27], int x, int y, int z)
{
    int block_id = 13;
    if(z < 0) {block_id -= 1;z += block_zsize;}
    if(z >= block_zsize) {block_id += 1;z -= block_zsize;}
    if(y < 0) {block_id -= 3;y += block_ysize;}
    if(y >= block_ysize) {block_id += 3;y -= block_ysize;}
    if(x < 0) {block_id -= 9;x += block_xsize;}
    if(x >= block_xsize) {block_id += 9;x -= block_xsize;}
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    const int entry = z + y * ystride + x * xstride;
    a_out = reinterpret_cast<T*>((unsigned long)a_array + (unsigned long)b[block_id])[entry];
    b_out = reinterpret_cast<T*>((unsigned long)b_array + (unsigned long)b[block_id])[entry];
}

template<typename T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__device__ void Lookup_Vector_3D(T out[3],const T* a[3],const T_offset_ptr b[27],int x,int y,int z)
{
    int block_id = 13;
    if(z < 0) {block_id -= 1;z += block_zsize;}
    if(z >= block_zsize) {block_id += 1;z -= block_zsize;}
    if(y < 0) {block_id -= 3;y += block_ysize;}
    if(y >= block_ysize) {block_id += 3;y -= block_ysize;}
    if(x < 0) {block_id -= 9;x += block_xsize;}
    if(x >= block_xsize) {block_id += 9;x -= block_xsize;}
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    const int entry = z + y * ystride + x * xstride;
    out[0] = reinterpret_cast<T*>((unsigned long)a[0] + (unsigned long)b[block_id])[entry];
    out[1] = reinterpret_cast<T*>((unsigned long)a[1] + (unsigned long)b[block_id])[entry];
    out[2] = reinterpret_cast<T*>((unsigned long)a[2] + (unsigned long)b[block_id])[entry];
}

template<class T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__global__ void Linear_Elasticity_Kernel_3D()
{
    enum{d = 3};
    enum{nodes_per_cell = 1 << d};
    //this kernel assumes we have more threads per block than entry per block
    enum{DATABLOCK = block_xsize * block_ysize * block_zsize,
         span = THREADBLOCK / DATABLOCK};
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    enum{padded_node_xsize = block_xsize + 2,
         padded_node_ysize = block_ysize + 2,
         padded_node_zsize = block_zsize + 2,
         padded_node_total_size = padded_node_xsize * padded_node_ysize * padded_node_zsize,
         padded_cell_xsize = block_xsize + 1,
         padded_cell_ysize = block_ysize + 1,
         padded_cell_zsize = block_zsize + 1,
         padded_cell_total_size = padded_cell_xsize * padded_cell_ysize * padded_cell_zsize};
    
    Parameters& para = *reinterpret_cast<Parameters*>(p_device);

    using T_BLOCK = const T (&)[block_xsize][block_ysize][block_zsize];

    static_assert(span == 1,"Only span of 1 is supported");
    static_assert(THREADBLOCK * 2 >= padded_node_total_size, "THREADBLOCK is too small!!");
    static_assert(THREADBLOCK * 2 >= padded_cell_total_size, "THREADBLOCK is too small!!");

    const unsigned int entry = threadIdx.x;
    const int z = entry % block_zsize;
    const int y = entry / block_zsize % block_ysize;
    const int x = entry / block_zsize / block_ysize;
    
    const int cell_z1 = (threadIdx.x % padded_cell_zsize);
    const int cell_y1 = (threadIdx.x / padded_cell_zsize % padded_cell_ysize);
    const int cell_x1 = (threadIdx.x / padded_cell_zsize / padded_cell_ysize);

    const int cell_z2 = ((threadIdx.x + THREADBLOCK) % padded_cell_zsize);
    const int cell_y2 = ((threadIdx.x + THREADBLOCK) / padded_cell_zsize % padded_cell_ysize);
    const int cell_x2 = ((threadIdx.x + THREADBLOCK) / padded_cell_zsize / padded_cell_ysize);

    const int node_z1 = (threadIdx.x % padded_node_zsize);
    const int node_y1 = (threadIdx.x / padded_node_zsize % padded_node_ysize);
    const int node_x1 = (threadIdx.x / padded_node_zsize / padded_node_ysize);

    const int node_z2 = (threadIdx.x + THREADBLOCK) % padded_node_zsize;
    const int node_y2 = (threadIdx.x + THREADBLOCK) / padded_node_zsize % padded_node_ysize;
    const int node_x2 = (threadIdx.x + THREADBLOCK) / padded_node_zsize / padded_node_ysize;

    __shared__ T_offset_ptr block_index[27];
    if(threadIdx.x < 27)
        block_index[threadIdx.x] = para.b[blockIdx.x * 27 + threadIdx.x];

    using T_K = const T (&)[2][2][2][2][2][2][d][d];
    T_K K_mu_converted = reinterpret_cast<T_K>(K_mu_device[0]);
    T_K K_la_converted = reinterpret_cast<T_K>(K_la_device[0]);
    if(blockIdx.x < para.number_of_blocks){
        __syncthreads();
        __shared__ T u_local[padded_node_xsize][padded_node_ysize][padded_node_zsize][d];
        __shared__ T mu_local[padded_cell_xsize][padded_cell_ysize][padded_cell_zsize];
        __shared__ T la_local[padded_cell_xsize][padded_cell_ysize][padded_cell_zsize];
        Lookup_Two_Values_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(mu_local[cell_x1][cell_y1][cell_z1],
                                                                                 la_local[cell_x1][cell_y1][cell_z1],
                                                                                 para.mu, para.lambda, block_index, 
                                                                                 cell_x1 - 1, cell_y1 - 1, cell_z1 - 1);
        if((threadIdx.x + THREADBLOCK) < padded_cell_total_size)
            Lookup_Two_Values_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(mu_local[cell_x2][cell_y2][cell_z2],
                                                                                     la_local[cell_x2][cell_y2][cell_z2],
                                                                                     para.mu, para.lambda, block_index, 
                                                                                     cell_x2 - 1, cell_y2 - 1, cell_z2 - 1);
        Lookup_Vector_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(u_local[node_x1][node_y1][node_z1], 
                                                                             para.u, block_index,
                                                                             node_x1 - 1, node_y1 - 1, node_z1 - 1);
            
        if((threadIdx.x + THREADBLOCK) < padded_node_total_size)
            Lookup_Vector_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(u_local[node_x2][node_y2][node_z2], 
                                                                                 para.u, block_index,
                                                                                 node_x2 - 1, node_y2 - 1, node_z2 - 1);
        __syncthreads();
        T f_node[d] = {0,0,0};
        T K_tmp[9];
        using T_MATRIX = const T (&)[d][d];
        T_MATRIX m = reinterpret_cast<T_MATRIX>(K_tmp[0]);
        // -1, -1, -1 stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y][z] * (&K_mu_converted[1][1][1][0][0][0][0][0])[i] + 
                       la_local[x][y][z] * (&K_la_converted[1][1][1][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y][z][v];

        // -1, -1, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y][z    ] * (&K_mu_converted[1][1][1][0][0][1][0][0])[i] + 
                       la_local[x][y][z    ] * (&K_la_converted[1][1][1][0][0][1][0][0])[i] + 
                       mu_local[x][y][z + 1] * (&K_mu_converted[1][1][0][0][0][0][0][0])[i] + 
                       la_local[x][y][z + 1] * (&K_la_converted[1][1][0][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y][z + 1][v];

        // -1, -1, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y][z + 1] * (&K_mu_converted[1][1][0][0][0][1][0][0])[i] + 
                       la_local[x][y][z + 1] * (&K_la_converted[1][1][0][0][0][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y][z + 2][v];
        
        // -1, 0, -1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y    ][z] * (&K_mu_converted[1][1][1][0][1][0][0][0])[i] + 
                       la_local[x][y    ][z] * (&K_la_converted[1][1][1][0][1][0][0][0])[i] + 
                       mu_local[x][y + 1][z] * (&K_mu_converted[1][0][1][0][0][0][0][0])[i] + 
                       la_local[x][y + 1][z] * (&K_la_converted[1][0][1][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y + 1][z][v];

        // -1, 0, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y    ][z    ] * (&K_mu_converted[1][1][1][0][1][1][0][0])[i] + 
                       la_local[x][y    ][z    ] * (&K_la_converted[1][1][1][0][1][1][0][0])[i] + 
                       mu_local[x][y    ][z + 1] * (&K_mu_converted[1][1][0][0][1][0][0][0])[i] + 
                       la_local[x][y    ][z + 1] * (&K_la_converted[1][1][0][0][1][0][0][0])[i] +
                       mu_local[x][y + 1][z    ] * (&K_mu_converted[1][0][1][0][0][1][0][0])[i] + 
                       la_local[x][y + 1][z    ] * (&K_la_converted[1][0][1][0][0][1][0][0])[i] + 
                       mu_local[x][y + 1][z + 1] * (&K_mu_converted[1][0][0][0][0][0][0][0])[i] + 
                       la_local[x][y + 1][z + 1] * (&K_la_converted[1][0][0][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y + 1][z + 1][v];

        // -1, 0, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y    ][z + 1] * (&K_mu_converted[1][1][0][0][1][1][0][0])[i] + 
                       la_local[x][y    ][z + 1] * (&K_la_converted[1][1][0][0][1][1][0][0])[i] + 
                       mu_local[x][y + 1][z + 1] * (&K_mu_converted[1][0][0][0][0][1][0][0])[i] + 
                       la_local[x][y + 1][z + 1] * (&K_la_converted[1][0][0][0][0][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y + 1][z + 2][v];

        // -1, +1, -1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y + 1][z] * (&K_mu_converted[1][0][1][0][1][0][0][0])[i] + 
                       la_local[x][y + 1][z] * (&K_la_converted[1][0][1][0][1][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y + 2][z][v];

        // -1, +1, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y + 1][z    ] * (&K_mu_converted[1][0][1][0][1][1][0][0])[i] + 
                       la_local[x][y + 1][z    ] * (&K_la_converted[1][0][1][0][1][1][0][0])[i] + 
                       mu_local[x][y + 1][z + 1] * (&K_mu_converted[1][0][0][0][1][0][0][0])[i] + 
                       la_local[x][y + 1][z + 1] * (&K_la_converted[1][0][0][0][1][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y + 2][z + 1][v];

        // -1, +1, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x][y + 1][z + 1] * (&K_mu_converted[1][0][0][0][1][1][0][0])[i] + 
                       la_local[x][y + 1][z + 1] * (&K_la_converted[1][0][0][0][1][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x][y + 2][z + 2][v];


        // 0, -1, -1 stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y][z] * (&K_mu_converted[1][1][1][1][0][0][0][0])[i] + 
                       la_local[x    ][y][z] * (&K_la_converted[1][1][1][1][0][0][0][0])[i] +
                       mu_local[x + 1][y][z] * (&K_mu_converted[0][1][1][0][0][0][0][0])[i] +  
                       la_local[x + 1][y][z] * (&K_la_converted[0][1][1][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y][z][v];

        // 0, -1, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y][z    ] * (&K_mu_converted[1][1][1][1][0][1][0][0])[i] + 
                       la_local[x    ][y][z    ] * (&K_la_converted[1][1][1][1][0][1][0][0])[i] + 
                       mu_local[x    ][y][z + 1] * (&K_mu_converted[1][1][0][1][0][0][0][0])[i] + 
                       la_local[x    ][y][z + 1] * (&K_la_converted[1][1][0][1][0][0][0][0])[i] +
                       mu_local[x + 1][y][z    ] * (&K_mu_converted[0][1][1][0][0][1][0][0])[i] + 
                       la_local[x + 1][y][z    ] * (&K_la_converted[0][1][1][0][0][1][0][0])[i] + 
                       mu_local[x + 1][y][z + 1] * (&K_mu_converted[0][1][0][0][0][0][0][0])[i] + 
                       la_local[x + 1][y][z + 1] * (&K_la_converted[0][1][0][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y][z + 1][v];

        // 0, -1, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y][z + 1] * (&K_mu_converted[1][1][0][1][0][1][0][0])[i] + 
                       la_local[x    ][y][z + 1] * (&K_la_converted[1][1][0][1][0][1][0][0])[i] + 
                       mu_local[x + 1][y][z + 1] * (&K_mu_converted[0][1][0][0][0][1][0][0])[i] + 
                       la_local[x + 1][y][z + 1] * (&K_la_converted[0][1][0][0][0][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y][z + 2][v];

        // 0, 0, -1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y    ][z] * (&K_mu_converted[1][1][1][1][1][0][0][0])[i] + 
                       la_local[x    ][y    ][z] * (&K_la_converted[1][1][1][1][1][0][0][0])[i] + 
                       mu_local[x    ][y + 1][z] * (&K_mu_converted[1][0][1][1][0][0][0][0])[i] + 
                       la_local[x    ][y + 1][z] * (&K_la_converted[1][0][1][1][0][0][0][0])[i] + 
                       mu_local[x + 1][y    ][z] * (&K_mu_converted[0][1][1][0][1][0][0][0])[i] + 
                       la_local[x + 1][y    ][z] * (&K_la_converted[0][1][1][0][1][0][0][0])[i] + 
                       mu_local[x + 1][y + 1][z] * (&K_mu_converted[0][0][1][0][0][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z] * (&K_la_converted[0][0][1][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y + 1][z][v];

        // 0, 0, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y    ][z    ] * (&K_mu_converted[1][1][1][1][1][1][0][0])[i] + 
                       la_local[x    ][y    ][z    ] * (&K_la_converted[1][1][1][1][1][1][0][0])[i] + 
                       mu_local[x    ][y    ][z + 1] * (&K_mu_converted[1][1][0][1][1][0][0][0])[i] + 
                       la_local[x    ][y    ][z + 1] * (&K_la_converted[1][1][0][1][1][0][0][0])[i] +
                       mu_local[x    ][y + 1][z    ] * (&K_mu_converted[1][0][1][1][0][1][0][0])[i] + 
                       la_local[x    ][y + 1][z    ] * (&K_la_converted[1][0][1][1][0][1][0][0])[i] + 
                       mu_local[x    ][y + 1][z + 1] * (&K_mu_converted[1][0][0][1][0][0][0][0])[i] + 
                       la_local[x    ][y + 1][z + 1] * (&K_la_converted[1][0][0][1][0][0][0][0])[i] +
                       mu_local[x + 1][y    ][z    ] * (&K_mu_converted[0][1][1][0][1][1][0][0])[i] + 
                       la_local[x + 1][y    ][z    ] * (&K_la_converted[0][1][1][0][1][1][0][0])[i] + 
                       mu_local[x + 1][y    ][z + 1] * (&K_mu_converted[0][1][0][0][1][0][0][0])[i] + 
                       la_local[x + 1][y    ][z + 1] * (&K_la_converted[0][1][0][0][1][0][0][0])[i] +
                       mu_local[x + 1][y + 1][z    ] * (&K_mu_converted[0][0][1][0][0][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z    ] * (&K_la_converted[0][0][1][0][0][1][0][0])[i] + 
                       mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][0][0][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][0][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y + 1][z + 1][v];

        // 0, 0, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y    ][z + 1] * (&K_mu_converted[1][1][0][1][1][1][0][0])[i] + 
                       la_local[x    ][y    ][z + 1] * (&K_la_converted[1][1][0][1][1][1][0][0])[i] + 
                       mu_local[x    ][y + 1][z + 1] * (&K_mu_converted[1][0][0][1][0][1][0][0])[i] + 
                       la_local[x    ][y + 1][z + 1] * (&K_la_converted[1][0][0][1][0][1][0][0])[i] +
                       mu_local[x + 1][y    ][z + 1] * (&K_mu_converted[0][1][0][0][1][1][0][0])[i] + 
                       la_local[x + 1][y    ][z + 1] * (&K_la_converted[0][1][0][0][1][1][0][0])[i] + 
                       mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][0][0][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][0][0][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y + 1][z + 2][v];

        // 0, +1, -1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y + 1][z] * (&K_mu_converted[1][0][1][1][1][0][0][0])[i] + 
                       la_local[x    ][y + 1][z] * (&K_la_converted[1][0][1][1][1][0][0][0])[i] +
                       mu_local[x + 1][y + 1][z] * (&K_mu_converted[0][0][1][0][1][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z] * (&K_la_converted[0][0][1][0][1][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y + 2][z][v];

        // 0, +1, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y + 1][z    ] * (&K_mu_converted[1][0][1][1][1][1][0][0])[i] + 
                       la_local[x    ][y + 1][z    ] * (&K_la_converted[1][0][1][1][1][1][0][0])[i] + 
                       mu_local[x    ][y + 1][z + 1] * (&K_mu_converted[1][0][0][1][1][0][0][0])[i] + 
                       la_local[x    ][y + 1][z + 1] * (&K_la_converted[1][0][0][1][1][0][0][0])[i] +
                       mu_local[x + 1][y + 1][z    ] * (&K_mu_converted[0][0][1][0][1][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z    ] * (&K_la_converted[0][0][1][0][1][1][0][0])[i] + 
                       mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][0][1][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][0][1][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y + 2][z + 1][v];

        // 0, +1, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x    ][y + 1][z + 1] * (&K_mu_converted[1][0][0][1][1][1][0][0])[i] + 
                       la_local[x    ][y + 1][z + 1] * (&K_la_converted[1][0][0][1][1][1][0][0])[i] + 
                       mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][0][1][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][0][1][1][0][0])[i];


        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 1][y + 2][z + 2][v];


        // +1, -1, -1 stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y][z] * (&K_mu_converted[0][1][1][1][0][0][0][0])[i] + 
                       la_local[x + 1][y][z] * (&K_la_converted[0][1][1][1][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y][z][v];

        // +1, -1, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y][z    ] * (&K_mu_converted[0][1][1][1][0][1][0][0])[i] + 
                       la_local[x + 1][y][z    ] * (&K_la_converted[0][1][1][1][0][1][0][0])[i] + 
                       mu_local[x + 1][y][z + 1] * (&K_mu_converted[0][1][0][1][0][0][0][0])[i] + 
                       la_local[x + 1][y][z + 1] * (&K_la_converted[0][1][0][1][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y][z + 1][v];

        // +1, -1, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y][z + 1] * (&K_mu_converted[0][1][0][1][0][1][0][0])[i] + 
                       la_local[x + 1][y][z + 1] * (&K_la_converted[0][1][0][1][0][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y][z + 2][v];
        
        // +1, 0, -1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y    ][z] * (&K_mu_converted[0][1][1][1][1][0][0][0])[i] + 
                       la_local[x + 1][y    ][z] * (&K_la_converted[0][1][1][1][1][0][0][0])[i] + 
                       mu_local[x + 1][y + 1][z] * (&K_mu_converted[0][0][1][1][0][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z] * (&K_la_converted[0][0][1][1][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y + 1][z][v];

        // +1, 0, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y    ][z    ] * (&K_mu_converted[0][1][1][1][1][1][0][0])[i] + 
                       la_local[x + 1][y    ][z    ] * (&K_la_converted[0][1][1][1][1][1][0][0])[i] + 
                       mu_local[x + 1][y    ][z + 1] * (&K_mu_converted[0][1][0][1][1][0][0][0])[i] + 
                       la_local[x + 1][y    ][z + 1] * (&K_la_converted[0][1][0][1][1][0][0][0])[i] +
                       mu_local[x + 1][y + 1][z    ] * (&K_mu_converted[0][0][1][1][0][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z    ] * (&K_la_converted[0][0][1][1][0][1][0][0])[i] + 
                       mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][1][0][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][1][0][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y + 1][z + 1][v];

        // +1, 0, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y    ][z + 1] * (&K_mu_converted[0][1][0][1][1][1][0][0])[i] + 
                       la_local[x + 1][y    ][z + 1] * (&K_la_converted[0][1][0][1][1][1][0][0])[i] + 
                       mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][1][0][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][1][0][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y + 1][z + 2][v];

        // +1, +1, -1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y + 1][z] * (&K_mu_converted[0][0][1][1][1][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z] * (&K_la_converted[0][0][1][1][1][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y + 2][z][v];

        // +1, +1, 0  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y + 1][z    ] * (&K_mu_converted[0][0][1][1][1][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z    ] * (&K_la_converted[0][0][1][1][1][1][0][0])[i] + 
                       mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][1][1][0][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][1][1][0][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y + 2][z + 1][v];

        // +1, +1, +1  stencil
        for(int i = 0; i < 9;++i)
            K_tmp[i] = mu_local[x + 1][y + 1][z + 1] * (&K_mu_converted[0][0][0][1][1][1][0][0])[i] + 
                       la_local[x + 1][y + 1][z + 1] * (&K_la_converted[0][0][0][1][1][1][0][0])[i];

        for(int w = 0; w < d;++w)
        for(int v = 0; v < d;++v)
            f_node[w] += m[w][v] * u_local[x + 2][y + 2][z + 2][v];

        for(int v = 0;v < d;++v)
            reinterpret_cast<T*>((unsigned long)para.f[v] + (unsigned long)block_index[13])[entry] = f_node[v]; 
    }
}
//#####################################################################
// Constructor 3D
//#####################################################################
template <class T, int log2_struct,class T_offset_ptr>
Linear_Elasticity_CUDA_Optimized<T,log2_struct,3,T_offset_ptr>::Linear_Elasticity_CUDA_Optimized(T* const f_input[d],const T* const u_input[d],
                                                                                                 const T* const mu_input,const T* const lambda_input,
                                                                                                 const T_offset_ptr* const b_input,const int size_input)
    :mu(mu_input),lambda(lambda_input),b(b_input),size(size_input)
{
    for(int v=0;v<d;++v){f[v]=f_input[v];u[v]=u_input[v];}
}
//#####################################################################
// Function Run
//#####################################################################
template <class T,int log2_struct,class T_offset_ptr> 
void Linear_Elasticity_CUDA_Optimized<T,log2_struct,3,T_offset_ptr>::Run()
{
    if(!symbol_initialized){
        cudaMemcpyToSymbol(K_mu_device,&K_mu[0][0][0][0],576*sizeof(float),0,cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(K_la_device,&K_lambda[0][0][0][0],576*sizeof(float),0,cudaMemcpyHostToDevice);
        symbol_initialized = true;}
    cudaDeviceSynchronize();
    //std::cout << "Block size: " << block_xsize << " * " << block_ysize << " * " << block_xsize << std::endl;
    int number_of_cuda_blocks = size;
    if(number_of_cuda_blocks == 0) return;
    
    Parameters p;
    for(int v=0;v<d;++v){
        p.f[v] = f[v];
        p.u[v] = u[v];}
    p.mu = mu;
    p.lambda = lambda;
    p.b = b;
    p.number_of_blocks = size;
    cudaMemcpyToSymbol(p_device,(void*)&p,sizeof(Parameters),0,cudaMemcpyHostToDevice);
    
    auto cudaerror = cudaGetLastError();    
    if(cudaSuccess!=cudaerror){
        std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}

    Linear_Elasticity_Kernel_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>
        <<<number_of_cuda_blocks,THREADBLOCK,0>>>();
    cudaDeviceSynchronize();
    cudaerror = cudaGetLastError();
    if(cudaSuccess!=cudaerror){
        std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
}
//#####################################################################################################
template class Linear_Elasticity_CUDA_Optimized<float,3,3,unsigned int>;
