//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#include <iostream>
#include "Project_CUDA.h"
using namespace SPGrid;
#define THREADBLOCK 512

template<class T,class T_offset_ptr,int number_of_elements_per_block> 
__global__ void Project(T* const u,const unsigned* const flag,const unsigned flag_to_clear)
{
    static_assert(THREADBLOCK == number_of_elements_per_block,"Threadblock must equals to the number of elements per block.");
    enum{page_size = 4096u};
    T* out = reinterpret_cast<T*>((unsigned long)u + (unsigned long)page_size * blockIdx.x); 
    const unsigned* flag_array = reinterpret_cast<const unsigned*>((unsigned long)flag + (unsigned long)page_size * blockIdx.x); 
    if(flag_array[threadIdx.x] & flag_to_clear){
        out[threadIdx.x] = 0;}
}
//#####################################################################
// Constructor 3D
//#####################################################################
template <class T, int log2_struct,class T_offset_ptr>
Project_CUDA<T,log2_struct,3,T_offset_ptr>::Project_CUDA(T* const u_input,const unsigned* const flag_input,const int size_input,
                                                         const unsigned int flag_to_clear_input)
    :u(u_input),flag(flag_input),size(size_input),flag_to_clear(flag_to_clear_input)
{
}
//#####################################################################
// Function Run
//#####################################################################
template <class T,int log2_struct,class T_offset_ptr> 
    void Project_CUDA<T,log2_struct,3,T_offset_ptr>::Run()
{
    const int number_of_cuda_blocks = size;
    if(number_of_cuda_blocks == 0) return;
    Project<T,T_offset_ptr,number_of_elements>
        <<<number_of_cuda_blocks,THREADBLOCK>>>
        (u,flag,flag_to_clear);
    // cudaDeviceSynchronize();
    // cudaError err = cudaGetLastError();
    // if(cudaSuccess != err){
    //     std::cerr << "Error in Minus Laplace Helper. Msg: "<< cudaGetErrorString(err) << std::endl;
    //     abort();
    // }
}

//#####################################################################################################
template class Project_CUDA<float,3,3,unsigned int>;

