//#####################################################################
// Copyright (c) 2016, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#include "Inner_Product_CUDA.h"
using namespace SPGrid;

#define THREADBLOCK 1024

using namespace SPGrid;
//#####################################################################
// Kernel Norm_Kernel
//#####################################################################
template<class T,int number_of_elements_per_block,int page_size,class T_offset_ptr> 
__global__ void Inner_Product_Kernel(const T* const u1,const T* const u2,double* buffer,unsigned int number_of_blocks){
    const int span = THREADBLOCK / number_of_elements_per_block;
    const int block = threadIdx.x / number_of_elements_per_block;
    const unsigned long entry = threadIdx.x % number_of_elements_per_block;
    __shared__ double cache[THREADBLOCK];
    unsigned int block_id = blockIdx.x * span + block;
    unsigned long data_offset = ((unsigned long)block_id * page_size + entry * sizeof(T));
    if (block_id < number_of_blocks){
        cache[threadIdx.x] = (*(reinterpret_cast<T*>((unsigned long)u1 + data_offset))) * (*(reinterpret_cast<T*>((unsigned long)u2 + data_offset)));}
    else{cache[threadIdx.x] = 0;}
    __syncthreads();
    int size=THREADBLOCK;
    while(size!=1){
        size=size>>1;
        if(threadIdx.x<size) cache[threadIdx.x] += cache[threadIdx.x+size];
        __syncthreads();}
    buffer[blockIdx.x]=cache[0];
}
//#####################################################################
// Function Run
//#####################################################################
template <class T,int log2_struct,class T_offset_ptr> 
double Inner_Product_CUDA<T,log2_struct,3,T_offset_ptr>::Run()
{
    enum{number_of_elements_per_block = block_xsize * block_ysize * block_zsize};
    const int span = THREADBLOCK / number_of_elements_per_block;
    unsigned int number_of_cuda_blocks = (size % span) ? (size / span + 1):(size / span);
    if(number_of_cuda_blocks == 0) return 0;
    double* buffer = NULL;
    if(cudaMalloc((void**)&buffer,number_of_cuda_blocks * sizeof(double)) != cudaSuccess) abort();
    Inner_Product_Kernel<T,number_of_elements_per_block,page_size,T_offset_ptr>
        <<<number_of_cuda_blocks,THREADBLOCK,0>>>
        (u1,u2,buffer,size);
    double* host_buffer = NULL;
    if(cudaMallocHost((void**)&host_buffer,number_of_cuda_blocks * sizeof(double)) != cudaSuccess) abort();
    cudaMemcpy(host_buffer,buffer,number_of_cuda_blocks * sizeof(double),cudaMemcpyDeviceToHost);
    double result = 0;
    for(int i = 0;i < number_of_cuda_blocks;++i) result += host_buffer[i];
    cudaFree(buffer);
    cudaFreeHost(host_buffer);
    return result;
}
//#####################################################################
template class Inner_Product_CUDA<float,3,3,unsigned int>;
