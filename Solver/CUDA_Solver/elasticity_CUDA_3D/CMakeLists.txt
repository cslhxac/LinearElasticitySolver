SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOMPILE_WITHOUT_READ_WRITE_SUPPORT -L${CUDA_TOOLKIT_ROOT_DIR}/lib64" )

add_executable( linear_elasticity_test
  main.cpp 
  ../CUDA_Kernels/CG_SYSTEM_CUDA.cpp
  ../CUDA_Kernels/CG_VECTOR_CUDA.cpp
  )


TARGET_INCLUDE_DIRECTORIES( linear_elasticity_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../Common_Libraries )
TARGET_INCLUDE_DIRECTORIES( linear_elasticity_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../Public_Library )
TARGET_INCLUDE_DIRECTORIES( linear_elasticity_test PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include )


target_link_libraries( linear_elasticity_test
  ${Boost_LIBRARIES}
  cudart
  SPGrid_Core
  PhysBAM_Tools
  CUDA_KERNELS
)
