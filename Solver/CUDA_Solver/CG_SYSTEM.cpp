//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_SYSTEM
//#####################################################################
#include <iomanip>
#include <SPGrid/Tools/SPGrid_Threading_Helper.h>
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>
#include <Common_Tools/Math_Tools/RANGE_ITERATOR.h>
#include "CG_SYSTEM.h"
#include "CG_VECTOR.h"
#include "SPGrid_Linear_Elasticity.h"
using namespace SPGrid;
using namespace PhysBAM;
//#####################################################################
// Constructor
//#####################################################################
template<class T_STRUCT,class T,int d> CG_SYSTEM<T_STRUCT,T,d>::
CG_SYSTEM(T_FIELD mu_field_input,T_FIELD lambda_field_input,T dx_input,int size_input)
    :BASE(false,false),mu_field(mu_field_input),lambda_field(lambda_field_input),dx(dx_input),size(size_input){}
//#####################################################################
// Function Multiply
//#####################################################################
template<class T_STRUCT,class T,int d> void CG_SYSTEM<T_STRUCT,T,d>::
Multiply(const VECTOR_BASE& v,VECTOR_BASE& result) const
{
    auto v_field = CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v).field;
    auto result_field = CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(result).field;
    auto& blocks = CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(result).blocks;
    SPGRID_LINEAR_ELASTICITY<T_STRUCT,T,d> kernel(blocks,v_field,result_field,mu_field,lambda_field,dx);
    kernel.Run();
}
//#####################################################################
// Function Inner_Product
//#####################################################################
template<class T_STRUCT,class T,int d> 
double CG_SYSTEM<T_STRUCT,T,d>::
Inner_Product(const VECTOR_BASE& v1,const VECTOR_BASE& v2) const
{
    auto v1_field=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v1).field;
    auto v2_field=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v2).field;
    auto& blocks=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v1).blocks;
    // Take dot-product of hierarchy, use doubles for temporaries
    double sum=0;
    auto blocks_pair=blocks.Get_Blocks();
    for(int v=0;v<d;++v){
        auto v1=v1_field[v].Get_Const_Array();
        auto v2=v2_field[v].Get_Const_Array();
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                sum+=v1(offset)*v2(offset);}}}
    return sum;
}
//#####################################################################
// Function Convergence_Norm
//#####################################################################
template<class T_STRUCT,class T,int d> T CG_SYSTEM<T_STRUCT,T,d>::
Convergence_Norm(const VECTOR_BASE& v) const
{
    auto v_field=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v).field;
    auto& blocks=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v).blocks;
    // Take dot-product of hierarchy, use doubles for temporaries
    T max=0;
    auto blocks_pair=blocks.Get_Blocks();
    for(int v=0;v<d;++v){
        auto data=v_field[v].Get_Const_Array();
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                max=(fabs(data(offset))>max)?fabs(data(offset)):max;}}}
    return max;
}
//#####################################################################
// Function Project
//#####################################################################
template<class T_STRUCT,class T,int d> void CG_SYSTEM<T_STRUCT,T,d>::
Project(VECTOR_BASE& v) const
{
    LOG::SCOPE scope("Project");
    auto v_field=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v).field;
    auto& blocks=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(v).blocks;
    T_INDEX block_size=v_field[0].allocator->Block_Size().template Cast<T_INDEX>();
    auto nblocks=blocks.Get_Blocks().second;
    auto offsets=blocks.Get_Blocks().first;
    #pragma omp parallel for
    for(int b=0;b<nblocks;++b){
        unsigned long base_offset=offsets[b];
        T_INDEX base_index=T_MASK::LinearToCoord(base_offset).template Cast<T_INDEX>();
        for(RANGE_ITERATOR<d> iterator(RANGE<T_INDEX>(base_index,base_index+block_size-1));
            iterator.Valid();iterator.Next(),base_offset+=sizeof(T)){
            const T_INDEX& index=iterator.Index();
            if(index(1)==0) for(int v=0;v<d;++v){v_field[v].Get_Array()(base_offset)=0;}
            if(index(1)==size-1)for(int v=0;v<d;++v){v_field[v].Get_Array()(base_offset)=0;}}}
}
//#####################################################################
// Function Set_Boundary_Conditions
//#####################################################################
template<class T_STRUCT,class T,int d> void CG_SYSTEM<T_STRUCT,T,d>::
Set_Boundary_Conditions(VECTOR_BASE& x) const
{
    Project(x);
    // LOG::SCOPE scope("Set Boundary Condition");
    // auto x_field=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(x).field;
    // auto& blocks=CG_VECTOR<T_STRUCT,T,d>::Cg_Vector(x).blocks;
    // T_INDEX block_size=mu_field.allocator->Block_Size().template Cast<T_INDEX>();
    // auto nblocks=blocks.Get_Blocks().second;
    // auto offsets=blocks.Get_Blocks().first;
    // #pragma omp parallel for
    // for(int b=0;b<nblocks;++b){
    //     unsigned long base_offset=offsets[b];
    //     T_INDEX base_index=T_MASK::LinearToCoord(base_offset).template Cast<T_INDEX>();
    //     for(RANGE_ITERATOR<d> iterator(RANGE<T_INDEX>(base_index,base_index+block_size-1));
    //         iterator.Valid();iterator.Next(),base_offset+=sizeof(T)){
    //         const T_INDEX& index=iterator.Index();
    //         if(index(1)==0) for(int v=0;v<d;++v){x_field[v].Get_Array()(base_offset)=0;}
    //         if(index(1)==31){for(int v=1;v<d;++v){x_field[v].Get_Array()(base_offset)=0;}
    //             x_field[0].Get_Array()(base_offset)=dx;}}}
}
//#####################################################################
// Function Project_Nullspace
//#####################################################################
template<class T_STRUCT,class T,int d> void CG_SYSTEM<T_STRUCT,T,d>::  
Project_Nullspace(VECTOR_BASE& x) const
{
}
//#####################################################################
// Function Apply_Preconditioner
//#####################################################################
template<class T_STRUCT,class T,int d> void CG_SYSTEM<T_STRUCT,T,d>::  
Apply_Preconditioner(const VECTOR_BASE& r, VECTOR_BASE& z) const
{
    z=r;
}
//#####################################################################
template class CG_SYSTEM<ELASTICITY_STRUCT<float>,float,2>;
template class CG_SYSTEM<ELASTICITY_STRUCT<float>,float,3>;
