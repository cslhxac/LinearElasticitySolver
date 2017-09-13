//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#include <iostream>
#include "CG_VECTOR_ADD_CUDA.h"
using namespace SPGrid;
#define THREADBLOCK 512

template<class T,class T_offset_ptr,int number_of_elements_per_block> 
__global__ void Vector_Add(T* const u_out,const T* const u_in1,
                           const T* const u_in2,const T a,
                           const bool zero_second)
{
    static_assert(THREADBLOCK == number_of_elements_per_block,"Threadblock must equals to the number of elements per block.");
    enum{page_size = 4096u};
    T* out = reinterpret_cast<T*>((unsigned long)u_out + (unsigned long)page_size * blockIdx.x); 
    const T* in1 = reinterpret_cast<const T*>((unsigned long)u_in1 + (unsigned long)page_size * blockIdx.x); 
    const T* in2 = reinterpret_cast<const T*>((unsigned long)u_in2 + (unsigned long)page_size * blockIdx.x); 
    if(zero_second)
        out[threadIdx.x] = a * in1[threadIdx.x];
    else
        out[threadIdx.x] = a * in1[threadIdx.x] + in2[threadIdx.x];
}
//#####################################################################
// Constructor 3D
//#####################################################################
template <class T, int log2_struct,class T_offset_ptr>
CG_VECTOR_ADD_CUDA<T,log2_struct,3,T_offset_ptr>::CG_VECTOR_ADD_CUDA(T* const u_out_input,const T* const u_in1_input,
                                                                     const T* const u_in2_input,const T a_input,
                                                                     const int size_input)
    :u_out(u_out_input),u_in1(u_in1_input),u_in2(u_in2_input),a(a_input),size(size_input),zero_second(false)
{
}
//#####################################################################
// Constructor 3D
//#####################################################################
template <class T, int log2_struct,class T_offset_ptr>
CG_VECTOR_ADD_CUDA<T,log2_struct,3,T_offset_ptr>::CG_VECTOR_ADD_CUDA(T* const u_out_input,const T* const u_in1_input,
                                                                     const T a_input,const int size_input)
    :u_out(u_out_input),u_in1(u_in1_input),u_in2(NULL),a(a_input),size(size_input),zero_second(true)
{
}
//#####################################################################
// Function Run
//#####################################################################
template <class T,int log2_struct,class T_offset_ptr> 
    void CG_VECTOR_ADD_CUDA<T,log2_struct,3,T_offset_ptr>::Run()
{
    const int number_of_cuda_blocks = size;
    if(number_of_cuda_blocks == 0) return;
    Vector_Add<T,T_offset_ptr,number_of_elements>
        <<<number_of_cuda_blocks,THREADBLOCK>>>
        (u_out,u_in1,u_in2,a,zero_second);
    // cudaDeviceSynchronize();
    // cudaError err = cudaGetLastError();
    // if(cudaSuccess != err){
    //     std::cerr << "Error in Minus Laplace Helper. Msg: "<< cudaGetErrorString(err) << std::endl;
    //     abort();
    // }
}

//#####################################################################################################
template class CG_VECTOR_ADD_CUDA<float,3,3,unsigned int>;

