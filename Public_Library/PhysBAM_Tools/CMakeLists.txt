add_library( PhysBAM_Tools
  Krylov_Solvers/CONJUGATE_GRADIENT.cpp
  Krylov_Solvers/KRYLOV_SOLVER.cpp
  Krylov_Solvers/KRYLOV_SYSTEM_BASE.cpp
  Krylov_Solvers/KRYLOV_VECTOR_BASE.cpp
  )

TARGET_INCLUDE_DIRECTORIES( PhysBAM_Tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../  )
