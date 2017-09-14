//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_SYSTEM_CUDA
//#####################################################################
#include "CG_SYSTEM_CUDA.h"
#include "CG_VECTOR_CUDA.h"
#include "Linear_Elasticity_CUDA_Optimized.h"
#include "Norm_Helper_CUDA.h"
#include "Inner_Product_CUDA.h"
#include "Project_CUDA.h"
#include "../ELASTICITY_FLAGS.h"
using namespace SPGrid;
using namespace PhysBAM;
//#####################################################################
// Constructor
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
CG_SYSTEM_CUDA(const T* mu_input,const T* lambda_input,const unsigned* flags_input,const T_offset_ptr* blocks_input,T dx_input,unsigned int n_blocks_input)
    :BASE(false,false),mu(mu_input),lambda(lambda_input),flags(flags_input),blocks(blocks_input),dx(dx_input),n_blocks(n_blocks_input)
{}
//#####################################################################
// Function Multiply
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> void CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
Multiply(const VECTOR_BASE& v,VECTOR_BASE& result) const
{
    auto v_field = CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v).field;
    auto result_field = CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(result).field;
    //PHYSBAM_ASSERT(n_blocks == CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(result).number_of_blocks);
    //PHYSBAM_ASSERT(n_blocks == CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v).number_of_blocks);
    Linear_Elasticity_CUDA_Optimized<T,log2_struct,d,T_offset_ptr> helper(result_field,v_field,mu,lambda,blocks,n_blocks,dx);
    helper.Run();
}
//#####################################################################
// Function Inner_Product
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> double CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
Inner_Product(const VECTOR_BASE& v1,const VECTOR_BASE& v2) const
{
    auto v1_field = CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v1).field;
    auto v2_field = CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v2).field;
    //PHYSBAM_ASSERT(n_blocks == CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v1).number_of_blocks);
    //PHYSBAM_ASSERT(n_blocks == CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v2).number_of_blocks);
    // Take dot-product of hierarchy, use doubles for temporaries
    double sum = 0;
    for(int v = 0;v < d;++v){
        Inner_Product_CUDA<T,log2_struct,d,T_offset_ptr> helper(v1_field[v],v2_field[v],n_blocks);
        sum += helper.Run();}
    return sum;
}
//#####################################################################
// Function Convergence_Norm
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> T CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
Convergence_Norm(const VECTOR_BASE& v) const
{
    auto v_field = CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v).field;
    //PHYSBAM_ASSERT(n_blocks == CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v).number_of_blocks);
    T max = 0;
    for(int v = 0;v < d;++v){
        Norm_Helper_CUDA<T,log2_struct,d,T_offset_ptr> helper(v_field[v],n_blocks);
        const T tmp = helper.Run();
        max = (max > tmp) ? max : tmp;}
    return max;
}
//#####################################################################
// Function Project
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> void CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
Project(VECTOR_BASE& v) const
{
    auto v_field = CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v).field;
    //PHYSBAM_ASSERT(n_blocks == CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::Cg_Vector(v).number_of_blocks);
    for(int v=0;v<d;++v){
        Project_CUDA<T,log2_struct,d,T_offset_ptr> helper(v_field[v],flags,n_blocks,Elasticity_Node_Type_DirichletX<<v);
        helper.Run();}
}
//#####################################################################
// Function Set_Boundary_Conditions
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> void CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
Set_Boundary_Conditions(VECTOR_BASE& x) const
{
    Project(x);
}
//#####################################################################
// Function Project_Nullspace
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> void CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
Project_Nullspace(VECTOR_BASE& x) const
{
}
//#####################################################################
// Function Apply_Preconditioner
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> void CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr>::
Apply_Preconditioner(const VECTOR_BASE& r, VECTOR_BASE& z) const
{
    z = r;
}
//#####################################################################
template class CG_SYSTEM_CUDA<float,3,3,unsigned>;
