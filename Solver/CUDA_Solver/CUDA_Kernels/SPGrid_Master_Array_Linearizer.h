//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGRID_MASTER_ARRAY_LINEARIZER_H__
#define __SPGRID_MASTER_ARRAY_LINEARIZER_H__
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <array>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
namespace SPGrid{

template<class T,int log2_struct, int d,class T_offset_ptr> class SPGrid_Master_Array_Linearizer;
////////////////////////////////////////////////////////////////////////////////
//Basicly this class maps cpu mmaped spgrid onto a dense array on accelerator //
////////////////////////////////////////////////////////////////////////////////
template<class T,int log2_struct,class T_offset_ptr>
class SPGrid_Master_Array_Linearizer<T,log2_struct,3,T_offset_ptr>
{
public:
    typedef T_offset_ptr T_offset_ptr;
    enum{d=3};
    typedef SPGrid_Mask<log2_struct,NextLogTwo<sizeof(T)>::value,d> T_MASK;
    enum{page_size=4096u,
         block_xsize=1u<<T_MASK::block_xbits,
         block_ysize=1u<<T_MASK::block_ybits,
         block_zsize=1u<<T_MASK::block_zbits,
         elements_per_block=T_MASK::elements_per_block};
    T_offset_ptr deadbeef_block;
    std::vector<std::array<T_offset_ptr,27> > b;
    T_offset_ptr* b_device;
    mutable std::vector<char*> data;//this contains the linearized data!
    mutable std::vector<char*> data_device;
    mutable std::vector<char*> data_device_aux; //auxiliary channels are used for CG, and they do not require to be mapped to CPU memory 
    std::unordered_map<unsigned long,T_offset_ptr> offsets_map;
    std::vector<unsigned long> offsets_list;
    unsigned int data_buffer_size;
    unsigned int number_of_blocks;
    SPGrid_Master_Array_Linearizer():b_device(NULL){}
    SPGrid_Master_Array_Linearizer(const int nAllocators,const std::pair<const unsigned long*,unsigned>& blocks):b_device(NULL)
    {Initialize(nAllocators,blocks);}
    void Initialize(const int nAllocators,const std::pair<const unsigned long*,unsigned>& blocks){
        number_of_blocks=blocks.second;
        if((unsigned long)(number_of_blocks+1)*page_size>std::template numeric_limits<T_offset_ptr>::max()){
            std::cerr<<"Allocating more than the pointer type."<<std::endl;abort();}
        for(int i=0;i<data.size();++i) cudaFreeHost(data[i]);
        data.resize(nAllocators);
        b.resize(number_of_blocks);
        offsets_list.resize(number_of_blocks);
        offsets_map.reserve(number_of_blocks);
        data_buffer_size=(number_of_blocks+1)*page_size;

        deadbeef_block=number_of_blocks*page_size;
        for(int i=0;i<nAllocators;++i){
            auto cudaerror=cudaMallocHost(&data[i],data_buffer_size);
            if(cudaSuccess!=cudaerror){
                std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
            //initialize the deadbeef block
            std::memset(reinterpret_cast<void*>(reinterpret_cast<unsigned long>(data[i])+(unsigned long)deadbeef_block),0,page_size);}

        //populate the hashtable first
        for(int i=0;i<number_of_blocks;++i){
            offsets_map.insert({blocks.first[i],i*page_size});
            offsets_list[i]=blocks.first[i];}

        //populate the rest of the blocks
        for(int i=0;i<number_of_blocks;++i){
            unsigned long base_offset=blocks.first[i];
            std::array<T_offset_ptr,27> offset_tmp;
            std::array<unsigned long,27> key_tmp;
            key_tmp[0]  = T_MASK::Packed_Offset<-block_xsize,-block_ysize,-block_zsize>(base_offset);
            key_tmp[1]  = T_MASK::Packed_Offset<-block_xsize,-block_ysize,0           >(base_offset);
            key_tmp[2]  = T_MASK::Packed_Offset<-block_xsize,-block_ysize,+block_zsize>(base_offset);

            key_tmp[3]  = T_MASK::Packed_Offset<-block_xsize,0           ,-block_zsize>(base_offset);
            key_tmp[4]  = T_MASK::Packed_Offset<-block_xsize,0           ,0           >(base_offset);
            key_tmp[5]  = T_MASK::Packed_Offset<-block_xsize,0           ,+block_zsize>(base_offset);

            key_tmp[6]  = T_MASK::Packed_Offset<-block_xsize,+block_ysize,-block_zsize>(base_offset);
            key_tmp[7]  = T_MASK::Packed_Offset<-block_xsize,+block_ysize,0           >(base_offset);
            key_tmp[8]  = T_MASK::Packed_Offset<-block_xsize,+block_ysize,+block_zsize>(base_offset);

            key_tmp[9]  = T_MASK::Packed_Offset<0           ,-block_ysize,-block_zsize>(base_offset);
            key_tmp[10] = T_MASK::Packed_Offset<0           ,-block_ysize,0           >(base_offset);
            key_tmp[11] = T_MASK::Packed_Offset<0           ,-block_ysize,+block_zsize>(base_offset);

            key_tmp[12] = T_MASK::Packed_Offset<0           ,0           ,-block_zsize>(base_offset);
            key_tmp[13] = T_MASK::Packed_Offset<0           ,0           ,0           >(base_offset);
            key_tmp[14] = T_MASK::Packed_Offset<0           ,0           ,+block_zsize>(base_offset);

            key_tmp[15] = T_MASK::Packed_Offset<0           ,+block_ysize,-block_zsize>(base_offset);
            key_tmp[16] = T_MASK::Packed_Offset<0           ,+block_ysize,0           >(base_offset);
            key_tmp[17] = T_MASK::Packed_Offset<0           ,+block_ysize,+block_zsize>(base_offset);

            key_tmp[18] = T_MASK::Packed_Offset<+block_xsize,-block_ysize,-block_zsize>(base_offset);
            key_tmp[19] = T_MASK::Packed_Offset<+block_xsize,-block_ysize,0           >(base_offset);
            key_tmp[20] = T_MASK::Packed_Offset<+block_xsize,-block_ysize,+block_zsize>(base_offset);

            key_tmp[21] = T_MASK::Packed_Offset<+block_xsize,0           ,-block_zsize>(base_offset);
            key_tmp[22] = T_MASK::Packed_Offset<+block_xsize,0           ,0           >(base_offset);
            key_tmp[23] = T_MASK::Packed_Offset<+block_xsize,0           ,+block_zsize>(base_offset);

            key_tmp[24] = T_MASK::Packed_Offset<+block_xsize,+block_ysize,-block_zsize>(base_offset);
            key_tmp[25] = T_MASK::Packed_Offset<+block_xsize,+block_ysize,0           >(base_offset);
            key_tmp[26] = T_MASK::Packed_Offset<+block_xsize,+block_ysize,+block_zsize>(base_offset);

            for(int p=0;p<27;++p){
                auto got=offsets_map.find(key_tmp[p]);
                if(got!=offsets_map.end()){
                    offset_tmp[p]=(*got).second;
                }else{
                    offset_tmp[p]=deadbeef_block;}}
            b[i]=offset_tmp;}
        // Now, allocate memory on GPU
        for(int i=0;i<data_device.size();++i) cudaFree(data_device[i]);
        data_device.resize(nAllocators);
        for(int i=0;i<nAllocators;++i){
            auto cudaerror=cudaMalloc(&data_device[i],data_buffer_size);
            if(cudaSuccess!=cudaerror){
                std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}}

        if(b_device) cudaFree(b_device);
        auto cudaerror=cudaMalloc(&b_device,(size_t)27*number_of_blocks*sizeof(T_offset_ptr));
        if(cudaSuccess!=cudaerror){
            std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
        cudaerror=cudaMemcpy(b_device,&b[0][0],(size_t)27*number_of_blocks*sizeof(T_offset_ptr),cudaMemcpyHostToDevice);
        if(cudaSuccess!=cudaerror){
            std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}        
    }
    void Copy_Data_To_Linearizer(const void* spgrid_array_ptr,int allocator_id)
    {
        #pragma omp parallel for
        for(unsigned int i=0;i<offsets_list.size();++i){
            std::memcpy(reinterpret_cast<void*>((unsigned long)data[allocator_id]+(unsigned long)(i)*page_size),
                        reinterpret_cast<void*>((unsigned long)spgrid_array_ptr+offsets_list[i]),page_size);}
    }
    void Copy_Data_From_Linearizer(void* spgrid_array_ptr,int allocator_id) const
    {
        #pragma omp parallel for
        for(unsigned int i=0;i<offsets_list.size();++i){
            std::memcpy(reinterpret_cast<void*>((unsigned long)spgrid_array_ptr+offsets_list[i]),
                        reinterpret_cast<void*>((unsigned long)data[allocator_id]+(unsigned long)(i)*page_size),page_size);}
    }
    template<typename T_STRUCT>
    void Accumulate_Data_From_Linearizer(SPGrid_Allocator<T_STRUCT,d>& allocator,int allocator_id,T T_STRUCT::*field,const T c=1) const
    {
        unsigned long spgrid_array_ptr=(unsigned long)&allocator.Get_Array(field)(0);
        #pragma omp parallel for
        for(unsigned int i=0;i<offsets_list.size();++i){
            unsigned long field_offset=OffsetOfMember<T_STRUCT,T>(field)*elements_per_block;
            T* spgrid_ptr=reinterpret_cast<T*>(spgrid_array_ptr+offsets_list[i]);
            const T* linearized_ptr=reinterpret_cast<T*>((unsigned long)data[allocator_id]+(unsigned long)(i)*page_size+field_offset);
            for(int e=0;e<elements_per_block;++e)
                spgrid_ptr[e]+=c*linearized_ptr[e];}
    }
    void Copy_Data_To_Device(int allocator_id)
    {
        // TODO: async it
        auto cudaerror=cudaMemcpy(data_device[allocator_id],data[allocator_id],data_buffer_size,cudaMemcpyHostToDevice);
        if(cudaSuccess!=cudaerror){
            std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
    }
    void Copy_Data_From_Device(int allocator_id)
    {
        // TODO: async it
        auto cudaerror=cudaMemcpy(data[allocator_id],data_device[allocator_id],data_buffer_size,cudaMemcpyDeviceToHost);
        if(cudaSuccess!=cudaerror){
            std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
    }
    void Clear_Allocator(int allocator_id){
        auto cudaerror=cudaMemset(data_device[allocator_id],0,data_buffer_size);
        if(cudaSuccess!=cudaerror){
            std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
    }
    void Deallocate_Auxiliary_Data(){
        for(int i=0;i<data_device_aux.size();++i) cudaFree(data_device_aux[i]);
        data_device_aux.clear();        
    }
    void Allocate_Auxiliary_Data(int n_allocators){
        // This function will clear the memories that are allocated.
        for(int i=0;i<data_device_aux.size();++i) cudaFree(data_device_aux[i]);
        data_device_aux.resize(n_allocators);
        for(int i=0;i<data_device_aux.size();++i){
            auto cudaerror=cudaMalloc(&data_device_aux[i],data_buffer_size);
            if(cudaSuccess!=cudaerror){
                std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
            cudaerror=cudaMemset(data_device_aux[i],0,data_buffer_size);
            if(cudaSuccess!=cudaerror){
                std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}}
    }
    void Clear_Auxiliary_Data(){
        for(int i=0;i<data_device_aux.size();++i){
            auto cudaerror=cudaMemset(data_device_aux[i],0,data_buffer_size);
            if(cudaSuccess!=cudaerror){
                std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}}    
    }
    char* Get_Auxiliary_Channel(int i){
        return data_device_aux[i];
    }
};
}
#endif
