cmake_minimum_required(VERSION 2.8)

message(" ")
message("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
message("USE -DMACHINE=<machine> to select the machine configuration")
message("  using machine: ${MACHINE}")
message("  + set C++ compiler by defining the CXX environment variable")
message("  + set C compiler by defining the CC environment variable")
message("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
message(" ")

include(conf/${MACHINE}.cmake)

set(CMAKE_CXX_FLAGS "-O3 -qopenmp -fopenmp -std=c++0x -std=c++11 -march=native -xHost -DGEMMT_AVAILABLE")
set(CMAKE_C_FLAGS  "-O3 -qopenmp -fopenmp -std=c1x -march=native -xHost -DGEMMT_AVAILABLE")

set(ED_USE_ILP64 ${I64})


# append ILP64 flag
if(${ED_USE_ILP64})
set(CMAKE_CXX_FLAGS "-O3 -DMKL_ILP64 -qopenmp -fopenmp -std=c++0x -std=c++11")
set(CMAKE_C_FLAGS  "-O3 -DMKL_ILP64 -qopenmp -fopenmp -std=c1x")
message("Using ILP64")
message("CMAKE_FLAGS" ${CMAKE_CXX_FLAGS})
endif(${ED_USE_ILP64})

#PETS, SLEPC
include_directories( ${ED_PETSC_DIR}/include/ ${ED_SLEPC_DIR}/include/
${ED_PETSC_DIR}/${ED_PETSC_ARCH}/include/
${ED_SLEPC_DIR}/${ED_PETSC_ARCH}/include/ )

#BOOST
include_directories( ${ED_BOOST_ROOT} ${ED_BOOST_ROOT}/include )

# MPI
include_directories( ${ED_MPI_HOME}/include/ ${ED_MPI_INC_DIR} )
set(MPILIB ${MPI_LIB})
# HDF5
include_directories( ${ED_HDF5_HOME}/include/ )

link_directories(
        ${ED_PETSC_DIR}/${ED_PETSC_ARCH}/lib/
        ${ED_SLEPC_DIR}/${ED_PETSC_ARCH}/lib/
	${ED_PTHREAD_LIB}
        ${ED_MPI_HOME}/lib/
        ${ED_MPI_LIB_DIR}
        ${ED_HDF5_HOME}/lib/
        ${ED_LAPACK_LIB_DIR}
        ${ED_BOOST_LIB_DIR}
    )

### LAPACK
include_directories(${ED_LAPACK_INCLUDE} ${ED_CBLAS_INC_DIR})
link_directories(${ED_LAPACK_LIB_DIR})
include_directories(${ED_MKLROOT}/include/)
link_directories(${ED_MKLROOT}/lib/intel64/)

if(${ED_USE_ILP64})
set(LAPACKLIB ${ED_LAPACK_LIB64})
message(${ED_LAPACK_LIB_DIR} " using lapack: ${LAPACKLIB}")
else(${ED_USE_ILP64})
set(LAPACKLIB ${ED_LAPACK_LIB})
message(${ED_LAPACK_LIB_DIR} " using lapack: ${LAPACKLIB}")
endif(${ED_USE_ILP64})

# MKL
if(${ED_USE_MKL})
add_definitions(-DUSE_MKL)
message("Using MKL")
endif(${ED_USE_MKL})

### EXEC FILEs

add_executable(xxz_krylov ed_XXZ_krylov.cc)
target_link_libraries(xxz_krylov ${MPILIB} ${LAPACKLIB} slepc petsc)

add_executable(ising_krylov ed_Ising_krylov.cc)
target_link_libraries(ising_krylov ${MPILIB} ${LAPACKLIB} slepc petsc)

add_executable(xxz_krylov_correlations ed_XXZ_krylov_correlations.cc)
target_link_libraries(xxz_krylov_correlations ${MPILIB} ${LAPACKLIB} slepc petsc)

add_executable(circuit circuit.cc)
target_link_libraries(circuit ${MPILIB} ${LAPACKLIB} petsc)

add_executable(xxz_si_test ed_XXZ_si_test.cc)
target_link_libraries(xxz_si_test ${MPILIB} ${LAPACKLIB} slepc petsc)

add_executable(xxz_si_prod ed_XXZ_si_prod.cc)
target_link_libraries(xxz_si_prod ${MPILIB} ${LAPACKLIB} slepc petsc)

add_executable(xxz_si_resonances ed_XXZ_si_resonances.cc)
target_link_libraries(xxz_si_resonances ${MPILIB} ${LAPACKLIB} slepc petsc)

add_executable(xxz_si_resonances2 ed_XXZ_si_resonances2.cc)
target_link_libraries(xxz_si_resonances2 ${MPILIB} ${LAPACKLIB} slepc petsc)