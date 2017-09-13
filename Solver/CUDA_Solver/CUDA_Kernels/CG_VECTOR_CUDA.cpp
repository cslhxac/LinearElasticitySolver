//#####################################################################
// Copyright 2015, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_VECTOR_CUDA
//#####################################################################
#include "CG_VECTOR_CUDA.h"
#include "CG_VECTOR_ADD_CUDA.h"
using namespace PhysBAM;
using namespace SPGrid;
//#####################################################################
// operator+=
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> KRYLOV_VECTOR_BASE<T>& CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::
operator+=(const BASE& bv)
{
    //PHYSBAM_ASSERT(Cg_Vector(bv).blocks == this->blocks);
    //PHYSBAM_ASSERT(Cg_Vector(bv).number_of_blocks == this->number_of_blocks);
    for(int v=0;v<d;++v){
        auto d1=field[v];
        auto d2=Cg_Vector(bv).field[v];
        CG_VECTOR_ADD_CUDA<T,log2_struct,d,T_offset_ptr> helper(d1,d2,d1,1.0f,number_of_blocks);
        helper.Run();}
    return *this;
}
//#####################################################################
// operator-=
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> KRYLOV_VECTOR_BASE<T>& CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::
operator-=(const BASE& bv)
{
    //PHYSBAM_ASSERT(Cg_Vector(bv).blocks == this->blocks);
    //PHYSBAM_ASSERT(Cg_Vector(bv).number_of_blocks == this->number_of_blocks);
    for(int v=0;v<d;++v){
        auto d1=field[v];
        auto d2=Cg_Vector(bv).field[v];
        CG_VECTOR_ADD_CUDA<T,log2_struct,d,T_offset_ptr> helper(d1,d2,d1,-1.0f,number_of_blocks);
        helper.Run();}
    return *this;
}
//#####################################################################
// operator*=
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> KRYLOV_VECTOR_BASE<T>& CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::
operator*=(const T a)
{
    for(int v=0;v<d;++v){
        auto d1=field[v];
        CG_VECTOR_ADD_CUDA<T,log2_struct,d,T_offset_ptr> helper(d1,d1,a,number_of_blocks);
        helper.Run();}
    return *this;
}
//#####################################################################
// Function Copy
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> void CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::
Copy(const T c,const BASE& bv)
{
    //PHYSBAM_ASSERT(Cg_Vector(bv).blocks == this->blocks);
    //PHYSBAM_ASSERT(Cg_Vector(bv).number_of_blocks == this->number_of_blocks);
    for(int v=0;v<d;++v){
        auto d1=field[v];
        auto d2=Cg_Vector(bv).field[v];
        CG_VECTOR_ADD_CUDA<T,log2_struct,d,T_offset_ptr> helper(d1,d2,c,number_of_blocks);
        helper.Run();}
}
//#####################################################################
// Function Copy
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> void CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::
Copy(const T c1,const BASE& bv1,const BASE& bv2)
{
    //PHYSBAM_ASSERT(Cg_Vector(bv1).blocks == this->blocks);
    //PHYSBAM_ASSERT(Cg_Vector(bv1).number_of_blocks == this->number_of_blocks);
    //PHYSBAM_ASSERT(Cg_Vector(bv2).blocks == this->blocks);
    //PHYSBAM_ASSERT(Cg_Vector(bv2).number_of_blocks == this->number_of_blocks);
    for(int v=0;v<d;++v){
        auto d1=field[v];
        auto d2=Cg_Vector(bv1).field[v];
        auto d3=Cg_Vector(bv2).field[v];
        CG_VECTOR_ADD_CUDA<T,log2_struct,d,T_offset_ptr> helper(d1,d2,d3,c1,number_of_blocks);
        helper.Run();}
}
//#####################################################################
// Function Raw_Size
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> int CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::
Raw_Size() const
{
    //PHYSBAM_NOT_IMPLEMENTED();
}
//#####################################################################
// Function Raw_Get
//#####################################################################
template<class T,int log2_struct,int d,class T_offset_ptr> T& CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr>::
Raw_Get(int i)
{
    //PHYSBAM_NOT_IMPLEMENTED();
}
//#####################################################################
template class CG_VECTOR_CUDA<float,3,3,unsigned>;

