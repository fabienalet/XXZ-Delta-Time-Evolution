cmake_minimum_required(VERSION 3.21)

#include(conf/${MACHINE}.cmake)

# could be changed according to the exact compiler installed and its options
set(CMAKE_CXX_COMPILER "mpiicpc")
set(CMAKE_CC_COMPILER "mpiicc")
#set(CMAKE_CXX_COMPILER "mpicxx")
#set(CMAKE_CC_COMPILER "mpicc")
#set(CMAKE_CXX_FLAGS "-O3 -g")
#set(CMAKE_C_FLAGS  "-O3 -g")
set(CMAKE_CXX_FLAGS "-O3 -g -qopenmp -DGEMMT_AVAILABLE")
set(CMAKE_C_FLAGS  "-O3 -g -qopenmp -DGEMMT_AVAILABLE")


include_directories( $ENV{PETSC_DIR}/include $ENV{SLEPC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include $ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/include )

#BOOST
include_directories( $ENV{BOOST_ROOT}/include/ $ENV{BOOST_ROOT}/ $ENV{BOOST_DIR}/ $ENV{BOOST_DIR}/include )

# MPI
include_directories( ${MPI_ROOT}/include/ )

link_directories(
	$ENV{PETSC_DIR}/lib/
	$ENV{SLEPC_DIR}/lib/
	$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib
	$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib
	$ENV{MPI_ROOT}/lib/
    )

# MKL
### LAPACK from MKL

# MKL
set(ED_USE_MKL 0)
if(${ED_USE_MKL})
include_directories($ENV{MKLROOT}/include/)
link_directories($ENV{MKL_ROOT}/lib/intel64/)
add_definitions(-DUSE_MKL)
set(LAPACKLIB mkl_lapack95_lp64 mkl_scalapack_lp64 mkl_blacs_intelmpi_lp64 mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread )
message("Using MKL")
else()
include_directories($ENV{OPENBLAS_ROOT}/include)
link_directories($ENV{OPENBLAS_ROOT}/lib/)
include_directories($ENV{SCALAPACK_ROOT}/include)
link_directories($ENV{SCALAPACK_ROOT}/lib/)
set(LAPACKLIB scalapack openblas omp)
endif(${ED_USE_MKL})



add_executable(xxz_krylov ed_XXZ_krylov.cc)
target_link_libraries(xxz_krylov ${MPILIB} ${LAPACKLIB} slepc petsc)

add_executable(xxz_si_test ed_XXZ_si_test.cc)
target_link_libraries(xxz_si_test ${MPILIB} ${LAPACKLIB} slepc petsc)
