//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Linear_Elsaticity_CUDA_Optimized__
#define __Linear_Elsaticity_CUDA_Optimized__
#include <algorithm>
#include <SPGrid/Core/SPGrid_Mask.h>

namespace SPGrid{
template<class T,int log2_struct, int d,class T_offset_ptr> class Linear_Elasticity_CUDA_Optimized;


template<class T,int log2_struct,class T_offset_ptr>
class Linear_Elasticity_CUDA_Optimized<T,log2_struct,3,T_offset_ptr>
{
    enum{d=3};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T)>::value,d> T_MASK;
    T* f[d];         // output stream
    const T* u[d];   // input stream
    const T* const mu;
    const T* const lambda;
    const T_offset_ptr* const b;   // block offset stream
    const int size;     // number of blocks to process
    enum {
        block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        block_zsize = 1u << T_MASK::block_zbits
    };

public:
    explicit Linear_Elasticity_CUDA_Optimized(T* const f[d],const T* const u[d],
                                              const T* const mu,const T* const lambda,
                                              const T_offset_ptr* const b,const int size);
    
    void Run();
};

};
#endif
