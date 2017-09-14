#ifndef __FILE_PARSER__
#define __FILE_PARSER__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Blocks.h>
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>
#include "ELASTICITY_FLAGS.h"
#include "ELASTICITY_STRUCT.h"
#include <fstream>
#include <regex>
#include <vector>

using namespace PhysBAM;

namespace SPGrid{

template<class T,class T_STRUCT,int d> 
class File_Parser{
    typedef ELASTICITY_FIELD<T,T_STRUCT,d> T_FIELD;
    typedef SPGrid_Blocks<NextLogTwo<sizeof(T_STRUCT)>::value,d> T_BLOCKS;
    typedef SPGrid_Allocator<T_STRUCT,d> SPG_Allocator; 
    typedef typename SPG_Allocator::template Array<T>::type SPG_Data_Array_Type;
    typedef typename SPG_Allocator::Array<const T>::type SPG_Const_Data_Array_Type;
    typedef typename SPG_Allocator::Array<T>::mask T_MASK;
    enum{elements_per_block=T_MASK::elements_per_block};
    enum{block_xsize = 1u << T_MASK::block_xbits,
         block_ysize = 1u << T_MASK::block_ybits,
         block_zsize = 1u << T_MASK::block_zbits};

public:
    
    struct Material{
        T mu;
        T lambda;};
    std::vector<Material> materials;
    // This is the nodal grid size, equals to (cell grid size + 1).
    std_array<int,d> size;
 
    File_Parser(){static_assert(d==3,"only 3D is supported for the file reader");}
    
    void Parse_Basic_Info(const std::string file_name)
    {
        std::ifstream file;
        file.open(file_name, std::ios::in);
        std::string line;
        bool materials_parsed=false;
        bool size_parsed=false;
        while(std::getline(file,line)){
            if(line.compare(0,4,std::string("#mat"))==0){
                std::getline(file,line);
                int nMaterials=std::stoi(line);
                std::cout<<"Number of materials: "<<nMaterials<<std::endl;
                materials.resize(nMaterials);
                std::cout<<"The list is: "<<std::endl;
                for(int i=0;i<nMaterials;++i){
                    std::getline(file,line);
                    std::string::size_type sz;
                    materials[i].mu=std::stof(line,&sz);
                    materials[i].lambda=std::stof(line.substr(sz));
                    std::cout<<"\tmu : "<<materials[i].mu<<", lambda: "<<materials[i].lambda<<std::endl;}                
                materials_parsed=true;}
            if(line.compare(0,5,std::string("#grid"))==0){
                std::getline(file,line);
                std::string::size_type sz;                
                size(0)=std::stoi(line,&sz)+1;
                line=line.substr(sz);
                size(1)=std::stoi(line,&sz)+1;
                line=line.substr(sz);
                size(2)=std::stoi(line)+1;
                std::cout<<"Node grid size: "<<size<<std::endl;
                size_parsed=true;}
            if(materials_parsed&&size_parsed) break;}
        file.close();
    }
    void Activate_SPGrid(T_BLOCKS& blocks,const std_array<int,d>& padded_size)
    {
        //Mark the pages active
        for(int i=0;i<padded_size(0)/block_xsize;++i)
        for(int j=0;j<padded_size(1)/block_ysize;++j)
        for(int k=0;k<padded_size(2)/block_zsize;++k){
            std_array<int,d> block_min_corner(i*block_xsize,j*block_ysize,k*block_zsize);
            if(block_min_corner(0)<=size(0)&&block_min_corner(1)<=size(1)&&block_min_corner(2)<=size(2))
                blocks.MarkPageActive(T_MASK::Linear_Offset(std_array<int,d>(block_min_corner)));}        
        blocks.Refresh_Block_Offsets();
    }
    void Populate_SPGrid(const std::string& file_name,
                         ELASTICITY_FIELD<T,T_STRUCT,d> mu_field,
                         ELASTICITY_FIELD<T,T_STRUCT,d> lambda_field,
                         ELASTICITY_FIELD<T,T_STRUCT,d> flag_field,
                         ELASTICITY_FIELD<T,T_STRUCT,d> u_fields[d],
                         ELASTICITY_FIELD<T,T_STRUCT,d> f_fields[d])
    {
        auto flag=flag_field.Get_Array();
        // iterate through the nodes
        for(int i=0;i<size(0);++i)
        for(int j=0;j<size(1);++j)
        for(int k=0;k<size(2);++k){
            std_array<int,d> node_index(i,j,k);
            const unsigned long offset=T_MASK::Linear_Offset(node_index);
            const unsigned tmp=Elasticity_Node_Type_Active;
            flag(offset)=*reinterpret_cast<const T*>(&tmp);}

        std::ifstream file;
        file.open(file_name, std::ios::in);
        std::string line;
        while(std::getline(file,line)){
            if(line.compare(0,4,std::string("#vox"))==0){
                auto mu=mu_field.Get_Array();
                auto lambda=lambda_field.Get_Array();
                std::getline(file,line);
                std::stringstream stream(line);
                // iterate through the cells
                for(int i=0;i<size(0)-1;++i)
                for(int j=0;j<size(1)-1;++j)
                for(int k=0;k<size(2)-1;++k){
                    std_array<int,d> cell_index(i,j,k);
                    const unsigned long offset=T_MASK::Linear_Offset(cell_index);
                    int mat;
                    stream>>mat;
                    mu(offset)=materials[mat].mu;
                    lambda(offset)=materials[mat].lambda;}}
            if(line.compare(0,6,std::string("#psi_D"))==0){
                std::getline(file,line);
                int nDirichlet_Nodes=std::stoi(line);
                for(int i=0;i<nDirichlet_Nodes;++i){
                    std_array<int,d> node_index;
                    std::getline(file,line);
                    std::string::size_type sz;                
                    node_index(0)=std::stoi(line,&sz);
                    line=line.substr(sz);
                    node_index(1)=std::stoi(line,&sz);
                    line=line.substr(sz);
                    node_index(2)=std::stoi(line);
                    const unsigned long offset=T_MASK::Linear_Offset(node_index);
                    std::getline(file,line);
                    const int axis=std::stoi(line,&sz);
                    const T value=std::stof(line.substr(sz));
                    const unsigned tmp=(*reinterpret_cast<const unsigned*>(&flag(offset)))|(Elasticity_Node_Type_DirichletX<<axis);
                    flag(offset)=*reinterpret_cast<const T*>(&tmp);
                    u_fields[axis].Get_Array()(offset)=value;}}
            if(line.compare(0,2,std::string("#f"))==0){
                std::getline(file,line);
                int nNuemann_Nodes=std::stoi(line);
                for(int i=0;i<nNuemann_Nodes;++i){
                    std_array<int,d> node_index;
                    std::getline(file,line);
                    std::string::size_type sz;                
                    node_index(0)=std::stoi(line,&sz);
                    line=line.substr(sz);
                    node_index(1)=std::stoi(line,&sz);
                    line=line.substr(sz);
                    node_index(2)=std::stoi(line);
                    const unsigned long offset=T_MASK::Linear_Offset(node_index);
                    std::getline(file,line);
                    f_fields[0].Get_Array()(offset)=std::stof(line,&sz);
                    line=line.substr(sz);
                    f_fields[1].Get_Array()(offset)=std::stof(line,&sz);
                    line=line.substr(sz);
                    f_fields[2].Get_Array()(offset)=std::stof(line);}}
        }     
        file.close();
    }
};
}
#endif
