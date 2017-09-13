//#####################################################################
// Copyright 2017, Haixiang Liu
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_VECTOR_CUDA
//#####################################################################
#ifndef __CG_VECTOR_CUDA__
#define __CG_VECTOR_CUDA__

#include <PhysBAM_Tools/Krylov_Solvers/KRYLOV_VECTOR_BASE.h>

namespace PhysBAM{

template<class T,int log2_struct,int d,class T_offset_ptr> class CG_SYSTEM_CUDA;

template<class T,int log2_struct,int d,class T_offset_ptr>
class CG_VECTOR_CUDA:public KRYLOV_VECTOR_BASE<T>
{
    typedef KRYLOV_VECTOR_BASE<T> BASE;
    T* field[d];
    const int number_of_blocks;
public:
    CG_VECTOR_CUDA(T* field_input[d],int number_of_blocks_input)
        :number_of_blocks(number_of_blocks_input)
    {for(int v=0;v<d;++v) field[v]=field_input[v];}
    
    static const CG_VECTOR_CUDA& Cg_Vector(const BASE& base)
    {return dynamic_cast<const CG_VECTOR_CUDA&>(base);}

    static CG_VECTOR_CUDA& Cg_Vector(BASE& base)
    {return dynamic_cast<CG_VECTOR_CUDA&>(base);}

    friend class CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>;
//#####################################################################
    BASE& operator+=(const BASE& bv);
    BASE& operator-=(const BASE& bv);
    BASE& operator*=(const T a);
    void Copy(const T c,const BASE& bv);
    void Copy(const T c1,const BASE& bv1,const BASE& bv2);
    int Raw_Size() const;
    T& Raw_Get(int i);
//#####################################################################   
};
}
#endif
