//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __Project_CUDA__
#define __Project_CUDA__
#include <algorithm>
#include <SPGrid/Core/SPGrid_Mask.h>

namespace SPGrid{
template<class T,int log2_struct, int d,class T_offset_ptr> class Project_CUDA;

template<class T,int log2_struct,class T_offset_ptr>
class Project_CUDA<T,log2_struct,3,T_offset_ptr>
{
    enum{d=3};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T)>::value,d> T_MASK;
    T* const u;         // output stream
    const unsigned * const flag;
    const int size;     // number of blocks to process
    const unsigned flag_to_clear; 
    enum {
        block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        block_zsize = 1u << T_MASK::block_zbits,
        number_of_elements = block_xsize * block_ysize * block_zsize
    };

public:
    explicit Project_CUDA(T* const u,const unsigned* const flag,const int size,const unsigned int flag_to_clear);    
    void Run();
};

}
#endif
