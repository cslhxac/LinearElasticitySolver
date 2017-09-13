//#####################################################################
// Copyright 2017, Haixiang Liu
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_VECTOR
//#####################################################################
#ifndef __CG_VECTOR__
#define __CG_VECTOR__

#include <PhysBAM_Tools/Krylov_Solvers/KRYLOV_VECTOR_BASE.h>
#include <PhysBAM_Tools/Grids_Uniform_Arrays/ARRAYS_ND.h>
#include <PhysBAM_Tools/Vectors/VECTOR_2D.h>
#include <PhysBAM_Tools/Vectors/VECTOR_3D.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Blocks.h>
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>
#include <SPGrid_Fluids/Simulation/FLUIDS_SIMULATION_DATA.h>
#include "ELASTICITY_STRUCT.h"

using namespace SPGrid;

namespace PhysBAM{

template<class T_STRUCT,class T,int d> class CG_SYSTEM;

template<class T_STRUCT,class T,int d>
class CG_VECTOR:public KRYLOV_VECTOR_BASE<T>
{
    typedef KRYLOV_VECTOR_BASE<T> BASE;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T>::type Const_data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const unsigned>::type Const_flag_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<unsigned>::type Flag_array_type;
    typedef SPGrid_Allocator<T_STRUCT,d> SPG_Allocator;
    typedef SPGrid_Blocks<NextLogTwo<sizeof(T_STRUCT)>::value,d> SPG_Blocks; 
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const unsigned>::mask T_MASK;
    typedef VECTOR<int,d> T_INDEX;
    typedef ELASTICITY_FIELD<T,T_STRUCT,d> T_FIELD;

    enum{ELEMENTS_PER_BLOCK=T_MASK::elements_per_block};
    
    T_FIELD field[d];
    SPG_Blocks& blocks;
public:
    CG_VECTOR(T_FIELD field_input[d],SPG_Blocks& blocks_input)
        :blocks(blocks_input)
    {for(int v=0;v<d;++v) field[v]=field_input[v];}
    
    static const CG_VECTOR& Cg_Vector(const BASE& base)
    {return dynamic_cast<const CG_VECTOR&>(base);}

    static CG_VECTOR& Cg_Vector(BASE& base)
    {return dynamic_cast<CG_VECTOR&>(base);}

    friend class CG_SYSTEM<T_STRUCT,T,d>;
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
