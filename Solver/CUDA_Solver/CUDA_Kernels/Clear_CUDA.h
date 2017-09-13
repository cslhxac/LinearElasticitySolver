//#####################################################################
// Copyright (c) 2016, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __CLEAR_CUDA__
#define __CLEAR_CUDA__
#include <algorithm>
#include <SPGrid/Core/SPGrid_Mask.h>

namespace SPGrid{
template<class T,int log2_struct, int d,class T_offset_ptr> class Clear_CUDA;

template<class T,int log2_struct,class T_offset_ptr>
class Clear_CUDA<T,log2_struct,3,T_offset_ptr>
{
    enum{d=3};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T)>::value,d> T_MASK;
    T* const u;                         // output stream
    const unsigned* const flag;         // input stream
    const unsigned int size;
    enum {
        page_size=4096,
        block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        block_zsize = 1u << T_MASK::block_zbits,
        number_of_elements = block_xsize * block_ysize * block_zsize
    };

public:
    explicit Clear_CUDA(T* const u_input,const unsigned* const flag_input,const unsigned int size_input)
        :u(u_input),flag(flag_input),size(size_input)
    {}
    explicit Clear_CUDA(T* const u_input,const unsigned int size_input)
        :u(u_input),flag(NULL),size(size_input)
    {}
    void Run();
};
}
#endif
