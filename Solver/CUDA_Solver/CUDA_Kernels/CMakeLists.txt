set (
    CUDA_HOST_COMPILER
    g++ 
)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=compute_52 -code=sm_52 --std=c++11 -Xcompiler "-fPIC" --maxrregcount 64 --use_fast_math --ptxas-options=-allow-expensive-optimizations=true,-O3
    )

CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../../../Common_Libraries )
  
CUDA_ADD_LIBRARY(CUDA_KERNELS
  Linear_Elasticity_CUDA_Optimized.cu
  Norm_Helper_CUDA.cu
  CG_VECTOR_ADD_CUDA.cu
  Inner_Product_CUDA.cu
  Project_CUDA.cu
  #Clear_CUDA.cu
  )
