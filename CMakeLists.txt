cmake_minimum_required(VERSION 2.8.0)

find_package(Boost 1.53 REQUIRED COMPONENTS filesystem program_options system)

SET(CPP11_FLAGS "-std=c++11")
SET(OPENMP_FLAGS "-fopenmp")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CPP11_FLAGS} ${OPENMP_FLAGS}" )
SET(CMAKE_CXX_COMPILER icc)

add_subdirectory( Solver/CUDA_Solver )
add_subdirectory( Public_Library/PhysBAM_Tools )
add_subdirectory( Common_Libraries )
