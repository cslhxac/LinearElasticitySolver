#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Blocks.h>
#include <SPGrid/Tools/SPGrid_Block_Iterator.h>
#include <chrono>
#include <random>
#include "../ELASTICITY_FLAGS.h"
#include "../ELASTICITY_STRUCT.h"
//#include "../HIERARCHICAL_RANGE_ITERATOR.h"
//#include "../SPGrid_Linear_Elasticity.h"
#include "../CUDA_Kernels/SPGrid_Master_Array_Linearizer.h"
#include "../CUDA_Kernels/Linear_Elasticity_CUDA_Optimized.h"
#include "../CUDA_Kernels/CG_SOLVER_CUDA.h"
//#include "../Write_SPGrid_To_Point_Cloud.h"
#include "../File_Parser.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace PhysBAM;
using namespace SPGrid;

int main(int argc,char* argv[]) {
    typedef float T;
    enum{d=3};
    typedef ELASTICITY_STRUCT<T> T_STRUCT;
    typedef SPGrid_Allocator<T_STRUCT,d> SPG_Allocator;
    typedef SPGrid_Blocks<NextLogTwo<sizeof(T_STRUCT)>::value,d> SPG_Blocks_Type;
    typedef typename SPG_Allocator::Array<T>::mask T_MASK;
    enum{elements_per_block=T_MASK::elements_per_block};

    po::options_description config;
    config.add_options()
        ( "help","Display helper manual")
        ( "input",po::value<std::string>(),"Input file name")
        ( "output",po::value<std::string>(),"Output file name")
        ( "dx",po::value<float>(),"Grid cell size")
        ( "GPU_ID",po::value<int>(),"Using GPU #")
        ;

    po::variables_map vm;po::store(po::parse_command_line(argc,argv,config),vm);
    po::notify(vm);

    if(vm.count("help")){
        std::cout<<config<<std::endl;
        return 1;}


    if(vm.count("GPU_ID")){
        cudaSetDevice(vm["GPU_ID"].as<int>());
    }else{
        cudaSetDevice(0);}

    std::string input_file;
    if(vm.count("input")){
        input_file=vm["input"].as<std::string>();
    }else{
        std::cout<<"Please provide an input file."<<std::endl;
        return 1;}

    std::string output_file="out";
    if(vm.count("output")){
        output_file=vm["output"].as<std::string>();}

    File_Parser<T,T_STRUCT,d> file_parser;
    file_parser.Parse_Basic_Info(input_file);

    T dx;
    if(vm.count("dx")){
        dx=vm["dx"].as<float>();
    }else{
        dx=1.0f/T(file_parser.size(0));}
    
    SPG_Allocator allocator1(file_parser.size),allocator2(file_parser.size),
        allocator3(file_parser.size),allocator4(file_parser.size),allocator5(file_parser.size);
    std_array<int,d> padded_size(allocator1.xsize_padded,allocator1.ysize_padded,allocator1.zsize_padded);
    std::cout<<"Padded (allocated) size : "<<padded_size<<std::endl;

    unsigned long padded_volume=padded_size(0);
    for(int v=1;v<d;v++) padded_volume*=padded_size(v);
    
    SPG_Blocks_Type blocks((unsigned long)padded_volume);
    file_parser.Activate_SPGrid(blocks,padded_size);
    
    ELASTICITY_FIELD<T,T_STRUCT,d> u_fields[d];
    ELASTICITY_FIELD<T,T_STRUCT,d> f_fields[d];
    u_fields[0].channel=&T_STRUCT::channel_0;u_fields[0].allocator=&allocator1;
    u_fields[1].channel=&T_STRUCT::channel_1;u_fields[1].allocator=&allocator1;
    u_fields[2].channel=&T_STRUCT::channel_0;u_fields[2].allocator=&allocator2;
    f_fields[0].channel=&T_STRUCT::channel_1;f_fields[0].allocator=&allocator2;
    f_fields[1].channel=&T_STRUCT::channel_0;f_fields[1].allocator=&allocator3;
    f_fields[2].channel=&T_STRUCT::channel_1;f_fields[2].allocator=&allocator3;
    ELASTICITY_FIELD<T,T_STRUCT,d> mu_field,lambda_field;
    mu_field.channel=&T_STRUCT::channel_0;mu_field.allocator=&allocator4;
    lambda_field.channel=&T_STRUCT::channel_1;lambda_field.allocator=&allocator4;
    ELASTICITY_FIELD<T,T_STRUCT,d> flag_field;
    flag_field.channel=&T_STRUCT::channel_0;flag_field.allocator=&allocator5;

    file_parser.Populate_SPGrid(input_file,mu_field,lambda_field,flag_field,u_fields,f_fields);

    //SPGrid_To_Point_Cloud<T,T_STRUCT,d> writer(blocks,u_fields,flag_field,"output",0,dx);
    
    using T_offset_ptr=unsigned int;
    CG_SOLVER_CUDA<T,T_STRUCT,d,T_offset_ptr> solver(blocks.Get_Blocks(),dx);
    solver.Set_U(allocator1,allocator2); 
    solver.Set_F(allocator2,allocator3); 
    solver.Set_Cell_Parameters(allocator4);
    solver.Set_Flags(allocator5);
    solver.Solve();
    solver.Update_U(allocator1,allocator2);        
    //Write to point cloud
    //SPGrid_To_Point_Cloud<T,T_STRUCT,d> writer(blocks,u_fields,flag_field,"output",1,dx);
    
    return 0;
}
