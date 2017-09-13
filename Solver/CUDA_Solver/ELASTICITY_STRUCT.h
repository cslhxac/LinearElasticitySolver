#ifndef __ELASTICITY_STRUCT_h__
#define __ELASTICITY_STRUCT_h__

#include <SPGrid/Core/SPGrid_Allocator.h>
template<typename T>
struct ELASTICITY_STRUCT
{
    T channel_0;
    T channel_1;
};

template<typename T,typename T_STRUCT,int d>
struct ELASTICITY_FIELD
{
    T ELASTICITY_STRUCT<T>::* channel;
    SPGrid::SPGrid_Allocator<T_STRUCT,d>* allocator;
    auto Get_Array() -> decltype(allocator->Get_Array(channel))
    {
        return allocator->Get_Array(channel);
    }
    auto Get_Const_Array() const -> decltype(allocator->Get_Const_Array(channel))
    {
        return allocator->Get_Const_Array(channel);
    }
};
#endif
