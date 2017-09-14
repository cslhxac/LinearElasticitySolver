#ifndef __WRITE_SPGRID_TO_POINT_CLOUD__
#define __WRITE_SPGRID_TO_POINT_CLOUD__
#include <Common_Tools/Math_Tools/RANGE_ITERATOR.h>
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Blocks.h>
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>
#include "ELASTICITY_FLAGS.h"
#include "ELASTICITY_STRUCT.h"
#include <fstream>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

using namespace PhysBAM;

namespace SPGrid{

template<class T,class T_STRUCT,int d> 
class SPGrid_To_Point_Cloud{
    typedef ELASTICITY_FIELD<T,T_STRUCT,d> T_FIELD;
    typedef SPGrid_Blocks<NextLogTwo<sizeof(T_STRUCT)>::value,d> T_BLOCKS;
    typedef SPGrid_Allocator<T_STRUCT,d> SPG_Allocator; 
    typedef typename SPG_Allocator::template Array<T>::type SPG_Data_Array_Type;
    typedef typename SPG_Allocator::Array<const T>::type SPG_Const_Data_Array_Type;
    typedef typename SPG_Allocator::Array<T>::mask T_MASK;
    enum{elements_per_block=T_MASK::elements_per_block};

public:
    SPGrid_To_Point_Cloud(T_BLOCKS& blocks,const T_FIELD u_fields[d],const T_FIELD flag_field,
                          const std::string output_directory,const int frame,const T dx)
    {
        static_assert(d==3,"Only 3D is supported by this write function");
        std::string file_directory=output_directory+"/"+std::to_string(frame);
        fs::create_directory(output_directory);
        fs::create_directory(file_directory);
        std::ofstream output;
        output.open(file_directory+"/point_cloud.ptc",std::ios::out|std::ios::binary);
        unsigned long n_nodes=0;
        auto flags=flag_field.Get_Const_Array();
        for(SPGrid_Block_Iterator<T_MASK> iterator(blocks.Get_Blocks());iterator.Valid();iterator.Next()){
            unsigned long offset=iterator.Offset();
            if((*reinterpret_cast<const unsigned*>(&flags(offset)))&Elasticity_Node_Type_Active) ++n_nodes;}
        
        output.write((char*)&n_nodes,sizeof(unsigned long));
        std::cout<<"Writing "<<n_nodes<<" number of nodes."<<std::endl;
        const std_array<unsigned int,d>& block_size=flag_field.allocator->Block_Size();
        for(SPGrid_Block_Iterator<T_MASK> iterator(blocks.Get_Blocks());iterator.Valid();iterator.Next_Block()){
            unsigned long offset=iterator.Offset();
            std_array<int,d> base_index=iterator.Index();
            for(int i=base_index(0);i<base_index(0)+block_size(0);++i)
            for(int j=base_index(1);j<base_index(1)+block_size(1);++j)
            for(int k=base_index(2);k<base_index(2)+block_size(2);++k,offset+=sizeof(T)){
                std_array<int,d> node_index(i,j,k);
                if((*reinterpret_cast<const unsigned*>(&flags(offset)))&Elasticity_Node_Type_Active){
                    std_array<T,d> u,X;
                    for(int v=0;v<d;++v) u(v)=u_fields[v].Get_Const_Array()(offset);                   
                    for(int v=0;v<d;++v) X(v)=T(node_index(v))*dx;
                    for(int v=0;v<d;++v) {const T x=X(v)+u(v);output.write((char*)&x,sizeof(T));}}}}

        for(SPGrid_Block_Iterator<T_MASK> iterator(blocks.Get_Blocks());iterator.Valid();iterator.Next_Block()){
            unsigned long offset=iterator.Offset();
            std_array<int,d> base_index=iterator.Index();
            for(int i=base_index(0);i<base_index(0)+block_size(0);++i)
            for(int j=base_index(1);j<base_index(1)+block_size(1);++j)
            for(int k=base_index(2);k<base_index(2)+block_size(2);++k,offset+=sizeof(T)){
                unsigned int node_type=0;
                if((*reinterpret_cast<const unsigned*>(&flags(offset)))&Elasticity_Node_Type_DirichletX||
                   (*reinterpret_cast<const unsigned*>(&flags(offset)))&Elasticity_Node_Type_DirichletY||
                   (*reinterpret_cast<const unsigned*>(&flags(offset)))&Elasticity_Node_Type_DirichletZ) {
                    node_type=2;
                    output.write((char*)&node_type,sizeof(unsigned int));}
                else if((*reinterpret_cast<const unsigned*>(&flags(offset)))&Elasticity_Node_Type_Active) {
                    node_type=1;
                    output.write((char*)&node_type,sizeof(unsigned int));}}}
        output.close();
    }
};
}
#endif
