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
# need to be adapted for other lapack's
include_directories($ENV{MKLROOT}/include/)
link_directories($ENV{MKL_ROOT}/lib/intel64/)
set(LAPACKLIB mkl_lapack95_lp64 mkl_scalapack_lp64 mkl_blacs_intelmpi_lp64 mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread )

set(ED_USE_MKL 1)
if(${ED_USE_MKL})
add_definitions(-DUSE_MKL)
message("Using MKL")
endif(${ED_USE_MKL})

add_executable(xxz_krylov ed_XXZ_krylov.cc)
target_link_libraries(xxz_krylov ${MPILIB} ${LAPACKLIB} slepc petsc)
