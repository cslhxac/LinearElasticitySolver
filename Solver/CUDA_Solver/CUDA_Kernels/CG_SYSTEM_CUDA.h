//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_SYSTEM_CUDA
//#####################################################################
#ifndef __CG_SYSTEM_CUDA__
#define __CG_SYSTEM_CUDA__

#include <PhysBAM_Tools/Krylov_Solvers/KRYLOV_SYSTEM_BASE.h>
#include "CG_VECTOR_CUDA.h"

namespace PhysBAM{

template<class T,int log2_struct,int d,class T_offset_ptr>
class CG_SYSTEM_CUDA:public KRYLOV_SYSTEM_BASE<T>
{
    typedef KRYLOV_SYSTEM_BASE<T> BASE;
    typedef KRYLOV_VECTOR_BASE<T> VECTOR_BASE;
    const T* mu;
    const T* lambda;
    const unsigned* flags;
    const T_offset_ptr* blocks;
    const T dx;
    const unsigned int n_blocks;    
//#####################################################################
public:
    CG_SYSTEM_CUDA(const T* mu,const T* lambda,const unsigned* flags,const T_offset_ptr* blocks,T dx,unsigned int n_blocks);
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
