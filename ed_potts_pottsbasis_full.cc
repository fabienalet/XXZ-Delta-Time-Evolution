static char help[] =
    "Shift-Invert ED for XXZ chain with Fibonnaci fields \n (C) Fabien Alet "
    "2017. \n\n"
    "The command line options are:\n"
    "  -L <L>, where <L> = size of the chain [default 6]\n"
    "  ( .... Description not complete ....) \n"
    "\n";

#include <omp.h>
#include <slepceps.h>
#include <boost/dynamic_bitset.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#else
typedef struct __MKL_Complex16 {
  double real;
  double imag;
} MKL_Complex16;
typedef int MKL_INT;
typedef MKL_Complex16 lapack_complex_double;

#include <cblas.h>
#include <lapacke.h>
#endif

double PI = acos(-1.);
using namespace std;

#include "Spin/Spin_parameters.h"

#include "Spin/Potts_basis.h"

#include "Spin/Potts_Pottsbasis.h"

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **argv) {
  cout.precision(20);
  SlepcInitialize(&argc, &argv, "slepc.options", help);
  Mat H;
  PetscErrorCode ierr;
  /**** Init parallel work ****/
  // For parallelization on node (openMP)
  int ENV_NUM_THREADS = mkl_get_max_threads();  /// Get value of OMP_NUM_THREADS
  omp_set_num_threads(
      ENV_NUM_THREADS);  // Deactivate OpenMP for PETSc and SLEPc
#ifdef USE_MKL
  mkl_set_num_threads(
      ENV_NUM_THREADS);  // Deactivate OpenMP for PETSc and SLEPc
#endif
  // For parallelization between nodes (MPI)
  int myrank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);


  int Q_=3;
  ierr = PetscOptionsGetInt(NULL, NULL, "-Q", &Q_, NULL);
  int number_of_states = Q_;

  int ZN_Sector=0;
  ierr = PetscOptionsGetInt(NULL, NULL, "-sector", &ZN_Sector, NULL);
  if (ZN_Sector>Q_) { exit(0); }


  Parameters myparameters(myrank);
  int L = myparameters.L;
  basis mybasis(L, ZN_Sector,myrank, number_of_states);
  //  mybasis.check_basis(); // ??
  int nconf = mybasis.total_number_of_confs;

  if (myrank == 0) {
    std::cout << "#L= " << L << " , Q=" << Q_ << " , Q_sector=" << ZN_Sector;
    std::cout << " number of states= " << nconf;
    std::cout << "\n";
  }

  PetscInt Istart, Iend;
  /************************** Hamiltonian **********************************/
  double *H_1d;
  H_1d = (double *)calloc(((long long int)nconf) * nconf, sizeof(double));
  Hamiltonian myHamiltonian(&mybasis, &H);
  myHamiltonian.get_parameters();
  myparameters.string_from_basis = mybasis.string_names;
  myparameters.string_from_H = myHamiltonian.string_names;
  myHamiltonian.create_matrix_lapack(H_1d);
  // perform full ED
  double w[nconf];
  bool compute_eigenvectors = 0; // myparameters.eigenvectors;
  myHamiltonian.diagonalize_lapack(H_1d, w, compute_eigenvectors);



  ofstream enout;
  ofstream rgapout;
  myparameters.init_filenames_energy(enout, rgapout, 1);

  // Do measurements
  double gap, previous_gap, previous_E;
  for (int i = 0; i < nconf; ++i) {
    // Energies and gap ratio
    if (i != 0) {
      double gap = w[i] - previous_E;
      if (i != 1) {
        double rg = gap / previous_gap;
        if (rg > 1) {
          rg = 1. / rg;
        }
        rgapout << rg << std::endl;
      }
      previous_gap = gap;
    }
    previous_E = w[i];
    enout << previous_E << std::endl;

  }
  rgapout.close();
  enout.close();

  SlepcFinalize();
  return 0;
}
