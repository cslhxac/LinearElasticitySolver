//#####################################################################
// Copyright (c) 2016, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#include "Clear_CUDA.h"
#include "../ELASTICITY_FLAGS.h"
using namespace SPGrid;

#define THREADBLOCK 512
using namespace SPGrid;
//#####################################################################
// Kernel Norm_Kernel
//#####################################################################
template<class T,class T_offset_ptr,int page_size,int number_of_elements_per_block> 
__global__ void Clear_Kernel(T* const u,const unsigned* const flag){
    static_assert(THREADBLOCK == number_of_elements_per_block,"Threadblock must equals to the number of elements per block.");
    T* out = reinterpret_cast<T*>((unsigned long)u + (unsigned long)page_size * blockIdx.x); 
    if(flag){
        const unsigned* flag_array = reinterpret_cast<const unsigned*>((unsigned long)flag + (unsigned long)page_size * blockIdx.x); 
        if(flag_array[threadIdx.x] & Elasticity_Node_Type_Dirichlet)
            out[threadIdx.x] = 0;
    }else
        out[threadIdx.x] = 0;
}
//#####################################################################
// Function Run
//#####################################################################
template <class T,int log2_struct,class T_offset_ptr> 
void Clear_CUDA<T,log2_struct,3,T_offset_ptr>::Run()
{
    const int number_of_cuda_blocks = size;
    if(number_of_cuda_blocks == 0) return;
    Clear_Kernel<T,T_offset_ptr,page_size,number_of_elements>
        <<<number_of_cuda_blocks,THREADBLOCK>>>
        (u,flag);
    // cudaDeviceSynchronize();
    // cudaError err = cudaGetLastError();
    // if(cudaSuccess != err){
    //     std::cerr << "Error in Minus Laplace Helper. Msg: "<< cudaGetErrorString(err) << std::endl;
    //     abort();
    // }
}
//#####################################################################################################
template class Clear_CUDA<float,3,3,unsigned int>;

