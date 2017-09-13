//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __CG_VECTOR_ADD_CUDA__
#define __CG_VECTOR_ADD_CUDA__
#include <algorithm>
#include <SPGrid/Core/SPGrid_Mask.h>

namespace SPGrid{
template<class T,int log2_struct, int d,class T_offset_ptr> class CG_VECTOR_ADD_CUDA;

template<class T,int log2_struct,class T_offset_ptr>
class CG_VECTOR_ADD_CUDA<T,log2_struct,3,T_offset_ptr>
{
    enum{d=3};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T)>::value,d> T_MASK;
    T* const u_out;         // output stream
    const T* const u_in1;   // input stream
    const T* const u_in2;   // input stream
    const T a;          // u_out = u_in1 * a + u_in2
    const int size;     // number of blocks to process
    const bool zero_second;
    enum {
        block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        block_zsize = 1u << T_MASK::block_zbits,
        number_of_elements = block_xsize * block_ysize * block_zsize
    };

public:
    explicit CG_VECTOR_ADD_CUDA(T* const u_out,const T* const u_in1,
                                const T* const u_in2,const T a,const int size);

    explicit CG_VECTOR_ADD_CUDA(T* const u_out,const T* const u_in1,
                                const T a,const int size);
    
    void Run();
};

}
#endif
