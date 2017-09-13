//#####################################################################
// Copyright 2015, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_VECTOR
//#####################################################################
#include "CG_VECTOR.h"
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>
#include <Common_Tools/Math_Tools/RANGE_ITERATOR.h>

using namespace PhysBAM;
using namespace SPGrid;
//#####################################################################
// operator+=
//#####################################################################
template<class T_STRUCT, class T,int d> KRYLOV_VECTOR_BASE<T>& CG_VECTOR<T_STRUCT,T,d>::
operator+=(const BASE& bv)
{
    for(int v=0;v<d;++v){
        auto d1=field[v].Get_Array();
        auto d2=Cg_Vector(bv).field[v].Get_Const_Array();
        auto blocks_pair=blocks.Get_Blocks();
        #pragma omp parallel for
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                d1(offset)+=d2(offset);}}}
    return *this;
}
//#####################################################################
// operator-=
//#####################################################################
template<class T_STRUCT, class T,int d> KRYLOV_VECTOR_BASE<T>& CG_VECTOR<T_STRUCT,T,d>::
operator-=(const BASE& bv)
{
    for(int v=0;v<d;++v){
        auto d1=field[v].Get_Array();
        auto d2=Cg_Vector(bv).field[v].Get_Const_Array();
        auto blocks_pair=blocks.Get_Blocks();
        #pragma omp parallel for
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                d1(offset)-=d2(offset);}}}
    return *this;
}
//#####################################################################
// operator*=
//#####################################################################
template<class T_STRUCT, class T,int d> KRYLOV_VECTOR_BASE<T>& CG_VECTOR<T_STRUCT,T,d>::
operator*=(const T a)
{
    for(int v=0;v<d;++v){
        auto d1=field[v].Get_Array();
        auto blocks_pair=blocks.Get_Blocks();
        #pragma omp parallel for
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                d1(offset)*=a;}}}
    return *this;
}
//#####################################################################
// Function Copy
//#####################################################################
template<class T_STRUCT, class T,int d> void CG_VECTOR<T_STRUCT,T,d>::
Copy(const T c,const BASE& bv)
{
    for(int v=0;v<d;++v){
        auto d1=field[v].Get_Array();
        auto d2=Cg_Vector(bv).field[v].Get_Const_Array();
        auto blocks_pair=blocks.Get_Blocks();
        #pragma omp parallel for
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                d1(offset)=c*d2(offset);}}}
}
//#####################################################################
// Function Copy
//#####################################################################
template<class T_STRUCT, class T,int d> void CG_VECTOR<T_STRUCT,T,d>::
Copy(const T c1,const BASE& bv1,const BASE& bv2)
{
    for(int v=0;v<d;++v){
        auto d1=field[v].Get_Array();
        auto d2=Cg_Vector(bv1).field[v].Get_Const_Array();
        auto d3=Cg_Vector(bv2).field[v].Get_Const_Array();
        auto blocks_pair=blocks.Get_Blocks();
        #pragma omp parallel for
        for(int b=0;b<blocks_pair.second;++b){
            unsigned long offset=blocks_pair.first[b];
            for(int e=0;e<ELEMENTS_PER_BLOCK;++e,offset+=sizeof(T)){
                d1(offset)=c1*d2(offset)+d3(offset);}}}
}
//#####################################################################
// Function Raw_Size
//#####################################################################
template<class T_STRUCT, class T,int d> int CG_VECTOR<T_STRUCT,T,d>::
Raw_Size() const
{
    PHYSBAM_NOT_IMPLEMENTED();
}
//#####################################################################
// Function Raw_Get
//#####################################################################
template<class T_STRUCT, class T,int d> T& CG_VECTOR<T_STRUCT,T,d>::
Raw_Get(int i)
{
    PHYSBAM_NOT_IMPLEMENTED();
}
//#####################################################################
template class CG_VECTOR<ELASTICITY_STRUCT<float>,float,2>;
template class CG_VECTOR<ELASTICITY_STRUCT<float>,float,3>;

