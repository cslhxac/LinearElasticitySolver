//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class SPGRID_LINEAR_ELASTICITY
//#####################################################################
#ifndef __SPGRID_LINEAR_ELASTICITY__
#define __SPGRID_LINEAR_ELASTICITY__
#include <Common_Tools/Math_Tools/RANGE_ITERATOR.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Blocks.h>
#include "Linear_Elasticity_System_Matrix.h"
namespace SPGrid{
using namespace PhysBAM;
template<typename T_STRUCT,typename T,int d>
class SPGRID_LINEAR_ELASTICITY
{
    typedef PhysBAM::VECTOR<int,d> T_INDEX;
    typedef PhysBAM::VECTOR<T,d> TV;
    typedef SPGrid_Allocator<T_STRUCT,d> SPG_Allocator; 
    typedef typename SPG_Allocator::Array<T>::mask T_MASK;
    typedef SPGrid_Blocks<NextLogTwo<sizeof(T_STRUCT)>::value,d> SPG_Blocks; 
    SPG_Blocks& blocks;
    ELASTICITY_FIELD<T,T_STRUCT,d> u_fields[d];
    ELASTICITY_FIELD<T,T_STRUCT,d> f_fields[d];
    ELASTICITY_FIELD<T,T_STRUCT,d> mu_field;
    ELASTICITY_FIELD<T,T_STRUCT,d> lambda_field;
    T dx;
    enum{NODES_PER_CELL=1<<d,ELEMENTS_PER_BLOCK=T_MASK::elements_per_block};
    enum{VERTICES_PER_CELL=1<<d};
    enum{PAGE_SIZE=4096u};
    // for the system matrix K
    // f_i(v) = Sum_{jw}(K(i)(j)(v)(w)*u_j(w))
    typedef VECTOR<VECTOR<MATRIX<T,d>,VERTICES_PER_CELL>,VERTICES_PER_CELL> SYSTEM_MATRIX;
    static void Apply_System_Matrix_Per_Cell(const SYSTEM_MATRIX& matrix, const T u_local[d][NODES_PER_CELL], T f_local[d][NODES_PER_CELL])
    {
        for(int n1=0;n1<NODES_PER_CELL;++n1){
            for(int v=0;v<d;++v) f_local[v][n1]=0;
            for(int n2=0;n2<NODES_PER_CELL;++n2){
                TV u,f;
                for(int v=0;v<d;++v) u(v+1)=u_local[v][n2];
                f=matrix(n1+1)(n2+1)*u;
                for(int v=0;v<d;++v) f_local[v][n1]+=f(v+1);}}
        
    }
public:
    SPGRID_LINEAR_ELASTICITY(SPG_Blocks& blocks_input,
                             const ELASTICITY_FIELD<T,T_STRUCT,d> u_fields_input[d],ELASTICITY_FIELD<T,T_STRUCT,d> f_fields_input[d],
                             const ELASTICITY_FIELD<T,T_STRUCT,d> mu_field_input,const ELASTICITY_FIELD<T,T_STRUCT,d> lambda_field_input,
                             T dx_input=1.0f)
        :blocks(blocks_input),mu_field(mu_field_input),lambda_field(lambda_field_input),dx(dx_input)
    {for(int v=0;v<d;++v) {u_fields[v]=u_fields_input[v];f_fields[v]=f_fields_input[v];}}
    
    void Run()
    {
        auto nblocks=blocks.Get_Blocks().second;
        auto offsets=blocks.Get_Blocks().first;
        unsigned long node_offsets[NODES_PER_CELL];
        int node=0;
        for(RANGE_ITERATOR<d> iterator(RANGE<T_INDEX>(T_INDEX(),T_INDEX::All_Ones_Vector()));iterator.Valid();iterator.Next())
            node_offsets[node++]=T_MASK::Linear_Offset(std_array<int,d>(iterator.Index()));
        auto mu=mu_field.Get_Const_Array();
        auto lambda=lambda_field.Get_Const_Array();
        T_INDEX block_size=mu_field.allocator->Block_Size().template Cast<T_INDEX>();
        #pragma omp parallel for
        for(int b=0;b<nblocks;++b){
            unsigned long base_offset=offsets[b];
            unsigned long max_offset=base_offset+ELEMENTS_PER_BLOCK*sizeof(T);
            //Clear the f channels
            for(unsigned long offset=base_offset;offset<max_offset;offset+=sizeof(T))
                for(int v=0;v<d;++v) f_fields[v].Get_Array()(offset)=0;
            T_INDEX base_index=T_MASK::LinearToCoord(base_offset).template Cast<T_INDEX>();
            for(RANGE_ITERATOR<d> cell_iterator(RANGE<T_INDEX>(base_index-T_INDEX::All_Ones_Vector(),base_index+block_size-1));
                cell_iterator.Valid();cell_iterator.Next()){
                unsigned long cell_offset=T_MASK::Linear_Offset(std_array<int,d>(cell_iterator.Index()));
                if(blocks.IsPageActive(cell_offset)&&(mu(cell_offset)||lambda(cell_offset))){
                    SYSTEM_MATRIX cell_matrix;
                    T u_local[d][NODES_PER_CELL];
                    T f_local[d][NODES_PER_CELL];
                    for(int node=0;node<NODES_PER_CELL;++node){
                        unsigned long node_of_cell_offset=T_MASK::Packed_Add(cell_offset,node_offsets[node]);
                        for(int v=0;v<d;++v) u_local[v][node]=u_fields[v].Get_Const_Array()(node_of_cell_offset);}
                    LINEAR_ELASTICITY_SYSTEM_MATRIX<T,d>::Create_Cell_Stiffness_Matrix(cell_matrix,dx,mu(cell_offset),lambda(cell_offset));
                    Apply_System_Matrix_Per_Cell(cell_matrix,u_local,f_local);   
                    for(int node=0;node<NODES_PER_CELL;++node){
                        unsigned long node_of_cell_offset=T_MASK::Packed_Add(cell_offset,node_offsets[node]);
                        if(node_of_cell_offset>=base_offset&&node_of_cell_offset<max_offset)
                            for(int v=0;v<d;++v) f_fields[v].Get_Array()(node_of_cell_offset)+=f_local[v][node];}}}}
    }
};
}
#endif
