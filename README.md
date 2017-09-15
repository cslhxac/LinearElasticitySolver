# LinearElasticitySolver

################################
         How to build
################################

1. create a build folder: midir build; cd build

2. cmake: cmake ../

3. It requires boost and cuda 7.0 to build.

4. If all goes right: make

################################
         How to run
################################

Solver/CUDA_Solver/elasticity_CUDA_3D/linear_elasticity_test  --help
  --help                Display helper manual
  --input arg           Input file name
  --output arg          Output file name
  --dx arg              Grid cell size
  --GPU_ID arg          Using GPU #
