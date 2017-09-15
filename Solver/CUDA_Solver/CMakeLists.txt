#set(CUDA_NVCC_EXECUTABLE /s/cuda-7.0.28/bin/nvcc)
#set(CUDA_INCLUDE_DIRS /s/cuda-7.0.28/include)
#set(CUDA_CUDART_LIBRARY /s/cuda-7.0.28/lib64)
#set(CUDA_TOOLKIT_ROOT_DIR /s/cuda-7.0.28)

find_package(CUDA 7.0 REQUIRED)

add_subdirectory( CUDA_Kernels )
add_subdirectory( elasticity_CUDA_3D )
