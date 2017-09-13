//#####################################################################
// Copyright 2017, Haixiang Liu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class CG_SOLVER_CUDA
//#####################################################################
#ifndef __CG_SOLVER_CUDA__
#define __CG_SOLVER_CUDA__

#include <PhysBAM_Tools/Krylov_Solvers/CONJUGATE_GRADIENT.h>
#include "CG_VECTOR_CUDA.h"
#include "CG_SYSTEM_CUDA.h"
#include "SPGrid_Master_Array_Linearizer.h"
#include "../ELASTICITY_STRUCT.h"
#include "../CUDA_Kernels/Linear_Elasticity_CUDA_Optimized.h"

using namespace SPGrid;

namespace PhysBAM{

template<class T,class T_STRUCT,int d,class T_offset_ptr> class CG_SOLVER_CUDA;
template<class T,class T_STRUCT,class T_offset_ptr>
class CG_SOLVER_CUDA<T,T_STRUCT,3,T_offset_ptr>
{    
    // The solver solves K::u = f.
    // With always zero Dirichlet condition.
public:
    enum{d=3,log2_struct=NextLogTwo<sizeof(T_STRUCT)>::value};
    typedef SPGrid_Allocator<T_STRUCT,d> SPG_Allocator;
    typedef typename SPG_Allocator::Array<T>::mask T_MASK;
    enum{elements_per_block=T_MASK::elements_per_block};
    const std::pair<const unsigned long*,unsigned>& blocks;
    SPGrid_Master_Array_Linearizer<T,log2_struct,d,T_offset_ptr> linearizer;
    // The linearizer has 5 allocators, they are 
    // allocator 0, channel 0: u_x
    // allocator 0, channel 1: u_y
    // allocator 1, channel 0: u_z
    // allocator 1, channel 1: f_x
    // allocator 2, channel 0: f_y
    // allocator 2, channel 1: f_z
    // allocator 3, channel 0: mu
    // allocator 3, channel 1: lambda
    // allocator 4, channel 0: flags
    // Please make sure the original SPGrid has the same structure.
    //--------------------------------------------------------------------//
    // All the following fields are on device.
    T* u[d];
    T* f[d];
    T* mu;
    T* lambda;
    T dx;
    unsigned* flags;
    CG_SOLVER_CUDA(const std::pair<const unsigned long*,unsigned>& blocks_input,T dx_input)
        :blocks(blocks_input),linearizer(5,blocks_input),dx(dx_input)
    {
        // Check for the cast here.
        // static_assert(sizeof(T)==sizeof(unsigned int))
        u[0]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device[0]
                                    +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        u[1]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device[0]
                                    +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        u[2]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device[1]
                                    +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        f[0]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device[1]
                                    +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        f[1]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device[2]
                                    +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        f[2]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device[2]
                                    +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        mu     = reinterpret_cast<T*>((unsigned long)linearizer.data_device[3]
                                    +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        lambda = reinterpret_cast<T*>((unsigned long)linearizer.data_device[3]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        flags  = reinterpret_cast<unsigned int*>((unsigned long)linearizer.data_device[4]
                                                 +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
    };
    void Set_U(SPG_Allocator& allocator1,SPG_Allocator& allocator2)
    {
        linearizer.Copy_Data_To_Linearizer(&allocator1.Get_Array(&T_STRUCT::channel_0)(0),0);
        linearizer.Copy_Data_To_Linearizer(&allocator2.Get_Array(&T_STRUCT::channel_0)(0),1);
        linearizer.Copy_Data_To_Device(0);
        linearizer.Copy_Data_To_Device(1);
    }
    void Set_F(SPG_Allocator& allocator1,SPG_Allocator& allocator2)
    {
        linearizer.Copy_Data_To_Linearizer(&allocator1.Get_Array(&T_STRUCT::channel_0)(0),1);
        linearizer.Copy_Data_To_Linearizer(&allocator2.Get_Array(&T_STRUCT::channel_0)(0),2);
        linearizer.Copy_Data_To_Device(1);
        linearizer.Copy_Data_To_Device(2);
    }
    void Set_Cell_Parameters(SPG_Allocator& allocator)
    {
        linearizer.Copy_Data_To_Linearizer(&allocator.Get_Array(&T_STRUCT::channel_0)(0),3);
        linearizer.Copy_Data_To_Device(3);
    }
    void Set_Flags(SPG_Allocator& allocator)
    {
        linearizer.Copy_Data_To_Linearizer(&allocator.Get_Array(&T_STRUCT::channel_0)(0),4);
        linearizer.Copy_Data_To_Device(4);
    }
    void Compute_F()
    {
        // Apply F = K::U
        Linear_Elasticity_CUDA_Optimized<T,log2_struct,d,T_offset_ptr> kernel(f,u,mu,lambda,
                                                                              linearizer.b_device,
                                                                              linearizer.number_of_blocks);
        kernel.Run();
    }
    void Solve()
    {
        linearizer.Allocate_Auxiliary_Data(5);
        T* r[d];//TODO: remove this
        T* q[d];
        T* s[d];
        r[0]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[0]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        r[1]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[0]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        r[2]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[1]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        q[0]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[1]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        q[1]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[2]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        q[2]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[2]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        s[0]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[3]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);
        s[1]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[3]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_1)*elements_per_block);
        s[2]   = reinterpret_cast<T*>((unsigned long)linearizer.data_device_aux[4]
                                      +(unsigned long)OffsetOfMember(&T_STRUCT::channel_0)*elements_per_block);

        CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr> u_v(u,blocks.second);
        CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr> f_v(f,blocks.second);
        CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr> r_v(r,blocks.second);
        CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr> q_v(q,blocks.second);
        CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr> s_v(s,blocks.second);
        CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr> k_v(f,blocks.second);// Not used
        CG_VECTOR_CUDA<T,log2_struct,d,T_offset_ptr> z_v(r,blocks.second);// Not used
        //Update Force Using Dirichlet Condition
        //Compute_F();
#if 1
        {
            // Apply F = K::U
            Linear_Elasticity_CUDA_Optimized<T,log2_struct,d,T_offset_ptr> kernel(r,u,mu,lambda,
                                                                                  linearizer.b_device,
                                                                                  linearizer.number_of_blocks);
            kernel.Run();
            f_v.Copy(-1,r_v,f_v);
        }
#endif
        //Zero initial guess, and clear dirichlet values.
        u_v.Copy(-1,u_v,u_v);
        
        CG_SYSTEM_CUDA<T,log2_struct,d,T_offset_ptr> cg_system(mu,lambda,flags,linearizer.b_device,dx,blocks.second);
        CONJUGATE_GRADIENT<T> cg;
        
        T norm=cg_system.Convergence_Norm(f_v);
        cg.print_residuals=true;
        cg.print_diagnostics=true;
        cg.Solve(cg_system,
                 u_v,
                 f_v,
                 q_v,
                 s_v,
                 r_v,
                 k_v,
                 z_v,
                 norm*1e-5,0,1000);

        // //Scale u_v with dx^3. Note now u_device hold the data for the correction du
        u_v.Copy(dx*dx*dx,u_v);
        linearizer.Copy_Data_From_Device(0);
        linearizer.Copy_Data_From_Device(1);
        linearizer.Deallocate_Auxiliary_Data();
    }
    void Update_U(SPG_Allocator& allocator1,SPG_Allocator& allocator2)
    {
        // Apply u -= du to SPGrid
        linearizer.template Accumulate_Data_From_Linearizer<T_STRUCT>(allocator1,0,&T_STRUCT::channel_0,-1.0f);
        linearizer.template Accumulate_Data_From_Linearizer<T_STRUCT>(allocator1,0,&T_STRUCT::channel_1,-1.0f);
        linearizer.template Accumulate_Data_From_Linearizer<T_STRUCT>(allocator2,1,&T_STRUCT::channel_0,-1.0f);
    }
};
}
#endif
