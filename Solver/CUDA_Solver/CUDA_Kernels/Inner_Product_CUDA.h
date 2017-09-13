//#####################################################################
// Copyright (c) 2016, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __INNER_PRODUCT_CUDA__
#define __INNER_PRODUCT_CUDA__
#include <algorithm>
#include <SPGrid/Core/SPGrid_Mask.h>

namespace SPGrid{
template<class T,int log2_struct, int d,class T_offset_ptr> class Inner_Product_CUDA;

template<class T,int log2_struct,class T_offset_ptr>
class Inner_Product_CUDA<T,log2_struct,3,T_offset_ptr>
{
    enum{d=3};
    typedef SPGrid_Mask<log2_struct, NextLogTwo<sizeof(T)>::value,d> T_MASK;
    const T* const u1;         // input stream
    const T* const u2;         // input stream
    const unsigned int size;
    enum {
        page_size=4096,
        block_xsize = 1u << T_MASK::block_xbits,
        block_ysize = 1u << T_MASK::block_ybits,
        block_zsize = 1u << T_MASK::block_zbits
    };

public:
    explicit Inner_Product_CUDA(const T* const u1_input,const T* const u2_input,const unsigned int size_input)
        :u1(u1_input),u2(u2_input),size(size_input)
    {}

    double Run();
};

}
#endif
