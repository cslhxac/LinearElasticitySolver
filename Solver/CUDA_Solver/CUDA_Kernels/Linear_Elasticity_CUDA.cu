//#####################################################################
// Copyright (c) 2017, Haixiang Liu
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#include <iostream>
#include <thrust/device_vector.h>
#include "Linear_Elasticity_CUDA.h"
using namespace SPGrid;
#define THREADBLOCK 512

__constant__ float K_mu_device[72];
__constant__ float K_la_device[72];
__constant__ float sign_device[24];
__constant__ int u_cell_offset[24];
__constant__ int k_mu_offset[72];

const float sign[8][3] = 
{
    {-(0 * 2 - 1),-(0 * 2 - 1), -(0 * 2 - 1)},
    {-(0 * 2 - 1),-(0 * 2 - 1), -(1 * 2 - 1)},
    {-(0 * 2 - 1),-(1 * 2 - 1), -(0 * 2 - 1)},
    {-(0 * 2 - 1),-(1 * 2 - 1), -(1 * 2 - 1)},
    {-(1 * 2 - 1),-(0 * 2 - 1), -(0 * 2 - 1)},
    {-(1 * 2 - 1),-(0 * 2 - 1), -(1 * 2 - 1)},
    {-(1 * 2 - 1),-(1 * 2 - 1), -(0 * 2 - 1)},
    {-(1 * 2 - 1),-(1 * 2 - 1), -(1 * 2 - 1)}        
};

//template <typename T,typename T_offset_ptr>
struct Parameters{
    float* f[3];
    const float* u[3];
    const float* mu;
    const float* lambda;
    const unsigned int* b;
    unsigned int number_of_blocks;
    const T dx;
};

// TODO: Remove the explicit template parameters on this one.
__constant__ char p_device[sizeof(Parameters)];

bool symbol_initialized = false;

// Index here: f_v = Sum_{iw}(K[v][w][i])
const float  __attribute__ ((aligned(32))) K_mu[3][3][8] =
{
    {
        {  32.f/72.f,  4.f/72.f,  4.f/72.f, -4.f/72.f, -8.f/72.f,-10.f/72.f,-10.f/72.f, -8.f/72.f},
        {   6.f/72.f,  3.f/72.f,  6.f/72.f,  3.f/72.f, -6.f/72.f, -3.f/72.f, -6.f/72.f, -3.f/72.f},
        {   6.f/72.f,  6.f/72.f,  3.f/72.f,  3.f/72.f, -6.f/72.f, -6.f/72.f, -3.f/72.f, -3.f/72.f}
    },
    {
        {   6.f/72.f,  3.f/72.f, -6.f/72.f, -3.f/72.f,  6.f/72.f,  3.f/72.f, -6.f/72.f, -3.f/72.f},
        {  32.f/72.f,  4.f/72.f, -8.f/72.f,-10.f/72.f,  4.f/72.f, -4.f/72.f,-10.f/72.f, -8.f/72.f},
        {   6.f/72.f,  6.f/72.f, -6.f/72.f, -6.f/72.f,  3.f/72.f,  3.f/72.f, -3.f/72.f, -3.f/72.f}
    },
    {
        {   6.f/72.f, -6.f/72.f,  3.f/72.f, -3.f/72.f,  6.f/72.f, -6.f/72.f,  3.f/72.f, -3.f/72.f},
        {   6.f/72.f, -6.f/72.f,  6.f/72.f, -6.f/72.f,  3.f/72.f, -3.f/72.f,  3.f/72.f, -3.f/72.f},
        {  32.f/72.f, -8.f/72.f,  4.f/72.f,-10.f/72.f,  4.f/72.f,-10.f/72.f, -4.f/72.f, -8.f/72.f}
    }
};

const float  __attribute__ ((aligned(32))) K_lambda[3][3][8] =
{
    {
        {   8.f/72.f,  4.f/72.f,  4.f/72.f,  2.f/72.f, -8.f/72.f, -4.f/72.f, -4.f/72.f, -2.f/72.f},
        {   6.f/72.f,  3.f/72.f, -6.f/72.f, -3.f/72.f,  6.f/72.f,  3.f/72.f, -6.f/72.f, -3.f/72.f},
        {   6.f/72.f, -6.f/72.f,  3.f/72.f, -3.f/72.f,  6.f/72.f, -6.f/72.f,  3.f/72.f, -3.f/72.f},
    },
    {
        {   6.f/72.f,  3.f/72.f,  6.f/72.f,  3.f/72.f, -6.f/72.f, -3.f/72.f, -6.f/72.f, -3.f/72.f},
        {   8.f/72.f,  4.f/72.f, -8.f/72.f, -4.f/72.f,  4.f/72.f,  2.f/72.f, -4.f/72.f, -2.f/72.f},
        {   6.f/72.f, -6.f/72.f,  6.f/72.f, -6.f/72.f,  3.f/72.f, -3.f/72.f,  3.f/72.f, -3.f/72.f},
    },
    {
        {   6.f/72.f,  6.f/72.f,  3.f/72.f,  3.f/72.f, -6.f/72.f, -6.f/72.f, -3.f/72.f, -3.f/72.f},
        {   6.f/72.f,  6.f/72.f, -6.f/72.f, -6.f/72.f,  3.f/72.f,  3.f/72.f, -3.f/72.f, -3.f/72.f},
        {   8.f/72.f, -8.f/72.f,  4.f/72.f, -4.f/72.f,  4.f/72.f, -4.f/72.f,  2.f/72.f, -2.f/72.f}
    }
};

template<class T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__device__ T Lookup_Value_3D(const T* a,const T_offset_ptr b[27],int x,int y,int z)
{
    int block_id = 13;
    if(z < 0) {block_id -= 1;z += block_zsize;}
    if(z >= block_zsize) {block_id += 1;z -= block_zsize;}
    if(y < 0) {block_id -= 3;y += block_ysize;}
    if(y >= block_ysize) {block_id += 3;y -= block_ysize;}
    if(x < 0) {block_id -= 9;x += block_xsize;}
    if(x >= block_xsize) {block_id += 9;x -= block_xsize;}
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    const int entry = z + y * ystride + x * xstride;
    return reinterpret_cast<T*>((unsigned long)a + (unsigned long)b[block_id])[entry];
}

template<class T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__device__ void Lookup_Two_Values_3D(T& a_out,T& b_out, const T* a_array, const T* b_array, const T_offset_ptr b[27], int x, int y, int z)
{
    int block_id = 13;
    if(z < 0) {block_id -= 1;z += block_zsize;}
    if(z >= block_zsize) {block_id += 1;z -= block_zsize;}
    if(y < 0) {block_id -= 3;y += block_ysize;}
    if(y >= block_ysize) {block_id += 3;y -= block_ysize;}
    if(x < 0) {block_id -= 9;x += block_xsize;}
    if(x >= block_xsize) {block_id += 9;x -= block_xsize;}
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    const int entry = z + y * ystride + x * xstride;
    a_out = reinterpret_cast<T*>((unsigned long)a_array + (unsigned long)b[block_id])[entry];
    b_out = reinterpret_cast<T*>((unsigned long)b_array + (unsigned long)b[block_id])[entry];
}

template<typename T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__device__ void Lookup_Vector_3D(T out[3],const T* a[3],const T_offset_ptr b[27],int x,int y,int z)
{
    int block_id = 13;
    if(z < 0) {block_id -= 1;z += block_zsize;}
    if(z >= block_zsize) {block_id += 1;z -= block_zsize;}
    if(y < 0) {block_id -= 3;y += block_ysize;}
    if(y >= block_ysize) {block_id += 3;y -= block_ysize;}
    if(x < 0) {block_id -= 9;x += block_xsize;}
    if(x >= block_xsize) {block_id += 9;x -= block_xsize;}
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    const int entry = z + y * ystride + x * xstride;
    out[0] = reinterpret_cast<T*>((unsigned long)a[0] + (unsigned long)b[block_id])[entry];
    out[1] = reinterpret_cast<T*>((unsigned long)a[1] + (unsigned long)b[block_id])[entry];
    out[2] = reinterpret_cast<T*>((unsigned long)a[2] + (unsigned long)b[block_id])[entry];
}

template<class T,class T_offset_ptr,int block_xsize,int block_ysize,int block_zsize> 
__global__ void Linear_Elasticity_Kernel_3D()
{
    enum{d = 3};
    enum{nodes_per_cell = 1 << d};
    //this kernel assumes we have more threads per block than entry per block
    enum{DATABLOCK = block_xsize * block_ysize * block_zsize,
         span = THREADBLOCK / DATABLOCK};
    enum{xstride = block_ysize * block_zsize,
         ystride = block_zsize,
         zstride = 1};
    enum{padded_node_xsize = block_xsize + 2,
         padded_node_ysize = block_ysize + 2,
         padded_node_zsize = block_zsize + 2,
         padded_node_total_size = padded_node_xsize * padded_node_ysize * padded_node_zsize,
         padded_cell_xsize = block_xsize + 1,
         padded_cell_ysize = block_ysize + 1,
         padded_cell_zsize = block_zsize + 1,
         padded_cell_total_size = padded_cell_xsize * padded_cell_ysize * padded_cell_zsize};
    
    Parameters& para = *reinterpret_cast<Parameters*>(p_device);

    using T_BLOCK = const T (&)[block_xsize][block_ysize][block_zsize];

    static_assert(span == 1,"Only span of 1 is supported");
    static_assert(THREADBLOCK * 2 >= padded_node_total_size, "THREADBLOCK is too small!!");
    static_assert(THREADBLOCK * 2 >= padded_cell_total_size, "THREADBLOCK is too small!!");

    const unsigned int entry = threadIdx.x;
    const int z = entry % block_zsize;
    const int y = entry / block_zsize % block_ysize;
    const int x = entry / block_zsize / block_ysize;
    
    const int cell_z1 = (threadIdx.x % padded_cell_zsize);
    const int cell_y1 = (threadIdx.x / padded_cell_zsize % padded_cell_ysize);
    const int cell_x1 = (threadIdx.x / padded_cell_zsize / padded_cell_ysize);

    const int cell_z2 = ((threadIdx.x + THREADBLOCK) % padded_cell_zsize);
    const int cell_y2 = ((threadIdx.x + THREADBLOCK) / padded_cell_zsize % padded_cell_ysize);
    const int cell_x2 = ((threadIdx.x + THREADBLOCK) / padded_cell_zsize / padded_cell_ysize);

    const int node_z1 = (threadIdx.x % padded_node_zsize);
    const int node_y1 = (threadIdx.x / padded_node_zsize % padded_node_ysize);
    const int node_x1 = (threadIdx.x / padded_node_zsize / padded_node_ysize);

    const int node_z2 = (threadIdx.x + THREADBLOCK) % padded_node_zsize;
    const int node_y2 = (threadIdx.x + THREADBLOCK) / padded_node_zsize % padded_node_ysize;
    const int node_x2 = (threadIdx.x + THREADBLOCK) / padded_node_zsize / padded_node_ysize;

    __shared__ T_offset_ptr block_index[27];
    if(threadIdx.x < 27)
        block_index[threadIdx.x] = para.b[blockIdx.x * 27 + threadIdx.x];

    using T_K_ROW = const T (&)[d][d][nodes_per_cell];
    T_K_ROW K_mu_converted = reinterpret_cast<T_K_ROW>(K_mu_device[0]);
    T_K_ROW K_la_converted = reinterpret_cast<T_K_ROW>(K_la_device[0]);
    if(blockIdx.x < para.number_of_blocks){
        __syncthreads();
        __shared__ T u_local[padded_node_xsize][padded_node_ysize][padded_node_zsize][d];
        __shared__ T mu_local[padded_cell_xsize][padded_cell_ysize][padded_cell_zsize];
        __shared__ T la_local[padded_cell_xsize][padded_cell_ysize][padded_cell_zsize];
        Lookup_Two_Values_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(mu_local[cell_x1][cell_y1][cell_z1],
                                                                                 la_local[cell_x1][cell_y1][cell_z1],
                                                                                 para.mu, para.lambda, block_index, 
                                                                                 cell_x1 - 1, cell_y1 - 1, cell_z1 - 1);
        if((threadIdx.x + THREADBLOCK) < padded_cell_total_size)
            Lookup_Two_Values_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(mu_local[cell_x2][cell_y2][cell_z2],
                                                                                     la_local[cell_x2][cell_y2][cell_z2],
                                                                                     para.mu, para.lambda, block_index, 
                                                                                     cell_x2 - 1, cell_y2 - 1, cell_z2 - 1);
        Lookup_Vector_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(u_local[node_x1][node_y1][node_z1], 
                                                                             para.u, block_index,
                                                                             node_x1 - 1, node_y1 - 1, node_z1 - 1);
            
        if((threadIdx.x + THREADBLOCK) < padded_node_total_size)
            Lookup_Vector_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>(u_local[node_x2][node_y2][node_z2], 
                                                                                 para.u, block_index,
                                                                                 node_x2 - 1, node_y2 - 1, node_z2 - 1);
        __syncthreads();
        T f_node[d] = {0,0,0};
        T u_local_mu[d][2][2][2];
        T u_local_la[d][2][2][2];
        // compute u_local_mu_{pql}[v] = Sum_{ijk}(mu_{ijk} * U_{pql}^{ijk}[v])
        // pql are the node indices of the cells, ijk are the cells sharing the current nodes
        // Note that U_{pql} are not in laxical graphical order rather than a mirrored version of of it,
        // base on the parity of the cell.
        for(int i = 0;i < 2;++i)
        for(int j = 0;j < 2;++j)
        for(int k = 0;k < 2;++k){
            const int sign[d] = {-(i * 2 - 1), -(j * 2 - 1), -(k * 2 - 1)};
            const T cell_mu = mu_local[x - i + 1][y - j + 1][z - k + 1];
            const T cell_la = la_local[x - i + 1][y - j + 1][z - k + 1];
#if 1
            for(int p = 0;p < 2;++p)
            for(int q = 0;q < 2;++q)
            for(int l = 0;l < 2;++l){
                const float* u_cell = u_local[x + p * sign[0] + 1][y + q * sign[1] + 1][z + l * sign[2] + 1];
                for(int v = 0;v < d;++v){
                    u_local_mu[v][p][q][l] = u_cell[v] * sign[v] * cell_mu;
                    u_local_la[v][p][q][l] = u_cell[v] * sign[v] * cell_la;}}
            
            // Now multiply the u_local_mu with the K matrix.
            using V_NODE_OF_CELL = const T (&)[d][nodes_per_cell];
            V_NODE_OF_CELL u_mu_converted = reinterpret_cast<V_NODE_OF_CELL>(u_local_mu[0][0][0]);
            V_NODE_OF_CELL u_la_converted = reinterpret_cast<V_NODE_OF_CELL>(u_local_la[0][0][0]);
            for(int v1 = 0;v1 < d;++v1)
            for(int v2 = 0;v2 < d;++v2)
            for(int node  = 0;node < nodes_per_cell;node++)
                f_node[v1] += sign[v1] * (u_mu_converted[v2][node] * K_mu_converted[v1][v2][node] + 
                                          u_la_converted[v2][node] * K_la_converted[v1][v2][node]);
#else
            // TODO: ?????????????????????WHY IS THIS 50% SLOWER THAN WHAT IS ABOVE?????????????
            float* u_cell[8];
            for(int node  = 0;node < nodes_per_cell;node++){
                u_cell[node] = u_local[x + ((node & 0x4) >> 2) * sign[0] + 1][y + ((node & 0x2) >> 1) * sign[1] + 1][z + (node & 0x1) * sign[2] + 1];
                for(int v1 = 0;v1 < d;++v1)
                for(int v2 = 0;v2 < d;++v2)
                    f_node[v1] += sign[v1] * (u_cell[node][v2] * sign[v2] * cell_mu * K_mu_converted[v1][v2][node] + 
                                              u_cell[node][v2] * sign[v2] * cell_la * K_la_converted[v1][v2][node]);}
#endif
        }
        for(int v = 0;v < d;++v)
            reinterpret_cast<T*>((unsigned long)para.f[v] + (unsigned long)block_index[13])[entry] = f_node[v] * p.dx; 
    }
}
//#####################################################################
// Constructor 3D
//#####################################################################
template <class T, int log2_struct,class T_offset_ptr>
Linear_Elasticity_CUDA<T,log2_struct,3,T_offset_ptr>::Linear_Elasticity_CUDA(T* const f_input[d],const T* const u_input[d],
                                                                             const T* const mu_input,const T* const lambda_input,
                                                                             const T_offset_ptr* const b_input,const int size_input,
                                                                             const T dx_input)
    :mu(mu_input),lambda(lambda_input),b(b_input),size(size_input),dx(dx_input)
{
    for(int v=0;v<d;++v){f[v]=f_input[v];u[v]=u_input[v];}
}
//#####################################################################
// Function Run
//#####################################################################
template <class T,int log2_struct,class T_offset_ptr> 
void Linear_Elasticity_CUDA<T,log2_struct,3,T_offset_ptr>::Run()
{
    if(!symbol_initialized){
        cudaMemcpyToSymbol(K_mu_device,&K_mu[0][0][0],72*sizeof(float),0,cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(K_la_device,&K_lambda[0][0][0],72*sizeof(float),0,cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(sign_device,&sign[0][0],24*sizeof(float),0,cudaMemcpyHostToDevice);
        symbol_initialized = true;}
    cudaDeviceSynchronize();
    //std::cout << "Block size: " << block_xsize << " * " << block_ysize << " * " << block_xsize << std::endl;
    int number_of_cuda_blocks = size;
    if(number_of_cuda_blocks == 0) return;
    
    Parameters p;
    for(int v=0;v<d;++v){
        p.f[v] = f[v];
        p.u[v] = u[v];}
    p.mu = mu;
    p.lambda = lambda;
    p.b = b;
    p.number_of_blocks = size;
    p.dx = dx;
    cudaMemcpyToSymbol(p_device,(void*)&p,sizeof(Parameters),0,cudaMemcpyHostToDevice);
    
    auto cudaerror = cudaGetLastError();    
    if(cudaSuccess!=cudaerror){
        std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}

    Linear_Elasticity_Kernel_3D<T,T_offset_ptr,block_xsize,block_ysize,block_zsize>
        <<<number_of_cuda_blocks,THREADBLOCK,0>>>();
#if 0
    cudaDeviceSynchronize();
    cudaerror = cudaGetLastError();
    if(cudaSuccess!=cudaerror){
        std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(cudaerror)<<std::endl;abort();}
#endif
}
//#####################################################################################################
template class Linear_Elasticity_CUDA<float,3,3,unsigned int>;
