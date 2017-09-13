//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_SYSTEM
//#####################################################################
#ifndef __CG_SYSTEM__
#define __CG_SYSTEM__

#include <PhysBAM_Tools/Krylov_Solvers/KRYLOV_SYSTEM_BASE.h>
#include <PhysBAM_Tools/Grids_Uniform_Arrays/ARRAYS_ND.h>
#include <PhysBAM_Tools/Vectors/VECTOR_2D.h>
#include <PhysBAM_Tools/Vectors/VECTOR_3D.h>
#include <PhysBAM_Tools/Vectors/VECTOR_ND.h>
#include <PhysBAM_Tools/Matrices/SPARSE_MATRIX_FLAT_NXN.h>
#include "CG_VECTOR.h"
#include "ELASTICITY_STRUCT.h"

using namespace SPGrid;

namespace PhysBAM{

template<class T_STRUCT,class T,int d> class NONLINEAR_ELASTICITY;

template<class T_STRUCT,class T,int d>
class CG_SYSTEM:public KRYLOV_SYSTEM_BASE<T>
{
    typedef KRYLOV_SYSTEM_BASE<T> BASE;
    typedef KRYLOV_VECTOR_BASE<T> VECTOR_BASE;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<T>::type Data_array_type;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T>::type Const_data_array_type;
    typedef SPGrid_Allocator<T_STRUCT,d> SPG_Allocator;
    typedef typename SPGrid_Allocator<T_STRUCT,d>::template Array<const T>::mask T_MASK;
    typedef VECTOR<int,d> T_INDEX;
    typedef ELASTICITY_FIELD<T,T_STRUCT,d> T_FIELD;

    enum{ELEMENTS_PER_BLOCK=T_MASK::elements_per_block};
    T_FIELD mu_field,lambda_field;
    T dx;
    int size;
//#####################################################################
public:
    CG_SYSTEM(T_FIELD mu_field,T_FIELD lambda_field,T dx,int size);
    void Multiply(const VECTOR_BASE& v,VECTOR_BASE& result) const;
    double Inner_Product(const VECTOR_BASE& x,const VECTOR_BASE& y) const;
    T Convergence_Norm(const VECTOR_BASE& x) const;
    void Project(VECTOR_BASE& x) const;
    void Set_Boundary_Conditions(VECTOR_BASE& x) const;
    void Project_Nullspace(VECTOR_BASE& x) const;    
protected:
    void Apply_Preconditioner(const VECTOR_BASE& r, VECTOR_BASE& z) const;
//#####################################################################
};
}
#endif
