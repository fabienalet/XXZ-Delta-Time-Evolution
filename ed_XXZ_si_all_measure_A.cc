static char help[] =
    "Shift-Invert ED for XXZ chain with Fibonnaci fields \n (C) Fabien Alet "
    "2017. \n\n"
    "The command line options are:\n"
    "  -L <L>, where <L> = size of the chain [default 6]\n"
    "  ( .... Description not complete ....) \n"
    "\n";

// #define PETSC_DESIRE_COMPLEX
// #define PETSC_USE_COMPLEX
// #include <complex>
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
#include <random>
#include <algorithm>
#include <iterator>
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

#include "Spin/Spin_basis.h"

#include "Spin/Spin_parameters.h"

#include "Spin/SpinOneHalfXXZ_disorder.h"

#include "Spin/Spin_observable.h"

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **argv) {
  cout.precision(20);
  int provided;
  PetscBool on_adastra=PETSC_FALSE;
  if (on_adastra) {
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  }
  SlepcInitialize(&argc, &argv, "slepc.options", help);

  /**** Init parallel work ****/
 // For parallelization on node (openMP)
 /*
  int ENV_NUM_THREADS=omp_get_num_threads();
  omp_set_num_threads(ENV_NUM_THREADS);
#ifdef USE_MKL
  ENV_NUM_THREADS=mkl_get_max_threads(); /// Get value of OMP_NUM_THREADS 
  mkl_set_num_threads(ENV_NUM_THREADS);
  omp_set_num_threads(ENV_NUM_THREADS);
#endif
*/
int ENV_NUM_THREADS=omp_get_max_threads();
  // For parallelization between nodes (MPI)
  int myrank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  /***** Petsc / Slepc data structure ***/
  Mat H;
  EPS eps;
  EPSType type;
  PetscErrorCode ierr;
  PetscReal error, tol, re, im;

  int number_of_states = 2;
  double Sz = 0;
  ierr = PetscOptionsGetReal(NULL, NULL, "-Sz", &Sz, NULL);
  CHKERRQ(ierr);

  Parameters myparameters(myrank);
  int L = myparameters.L;
  basis mybasis(L, Sz, myrank, number_of_states);
  //  mybasis.check_basis(); // ??
  int nconf = mybasis.total_number_of_confs;

  if (myrank == 0) {
    std::cout << "# L = " << L << " number of states=" << nconf << std::endl;
  }
  observable myobservable(&mybasis, ENV_NUM_THREADS);

  PetscInt Istart, Iend;
  /************************** Hamiltonian **********************************/
  {
    std::vector<int> local_block_sizes(mpisize, nconf / mpisize);
    for (size_t i = 0; i < nconf % mpisize; i++)
      local_block_sizes[i]++;  // distribute evenly
    Istart = 0;
    for (size_t i = 0; i < myrank; i++) Istart += local_block_sizes[i];
    Iend = Istart + local_block_sizes[myrank];

    Hamiltonian myHamiltonian(&mybasis, &H);
    myHamiltonian.get_parameters();

    myparameters.string_from_basis = mybasis.string_names;
    myparameters.string_from_H = myHamiltonian.string_names;

    //	 myobservable.get_local_field_x(myHamiltonian.local_field_x);
    // myobservable.get_local_field_z(myHamiltonian.local_field_z);
    myHamiltonian.init_matrix_sizes(Istart, Iend);
    ierr = MatSetUp(H);
    CHKERRQ(ierr);
    myHamiltonian.create_matrix(Istart, Iend);

    if (myrank == 0){ 
      myHamiltonian.print_local_fields();
    }

  }

  /****  Matrix assembly ****/
  MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
  MatSetOption(H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE);
  if (myrank == 0) std::cout << "# Assembly... " << std::flush;
  ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  if (myrank == 0) {
    std::cout << "Hamiltonian matrix assembly done." << std::endl;
  }

  // MatView(H,PETSC_VIEWER_STDOUT_WORLD);

  if (myrank == 0) {
    std::cout << "#L= " << L << " ,";
    std::cout << " number of states= " << nconf;
    std::cout << "\n";
  }

  PetscScalar Eminc, Emaxc;
  bool do_ev = 1;
  if (do_ev) {
    ierr = EPSCreate(PETSC_COMM_WORLD, &eps);
    CHKERRQ(ierr);
    ierr = EPSSetOperators(eps, H, NULL);
    CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_HEP);
    CHKERRQ(ierr);
    EPSSetDimensions(eps, 1, PETSC_DECIDE, PETSC_DECIDE);

    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    ierr = EPSSolve(eps);
    CHKERRQ(ierr);
    EPSGetEigenvalue(eps, 0, &Eminc, NULL);
    double Emin = PetscRealPart(Eminc);
    if (0 == myrank) std::cout << "Emin of H = " << Emin << "\n";
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
    ierr = EPSSolve(eps);
    CHKERRQ(ierr);
    EPSGetEigenvalue(eps, 0, &Emaxc, NULL);
    double Emax = PetscRealPart(Emaxc);
    if (0 == myrank) std::cout << "Emax of H = " << Emax << "\n";
  }
  /************* Measurement on eigenvectors *********/
  // Initialize file names

  Vec xr;
  MatCreateVecs(H, &xr, NULL);
  

  std::vector<double> targets;
  if (myparameters.target_infinite_temperature) {
    PetscScalar E_infinite_temperature;
    ierr = MatGetTrace(H, &E_infinite_temperature);
    CHKERRQ(ierr);
    E_infinite_temperature /= mybasis.total_number_of_confs;
    if (0 == myrank)
      std::cout << "E(beta = 0) = " << E_infinite_temperature << " E_renorm = "
                << (E_infinite_temperature - Eminc) / (Emaxc - Eminc)
                << std::endl;
    targets.push_back((E_infinite_temperature - Eminc) / (Emaxc - Eminc));
  } else {
    targets = myparameters.targets;
  }

  PetscBool special_energy_set=PETSC_FALSE;
  PetscReal special_energy=0.;
  PetscOptionsGetReal(NULL, NULL, "-special_energy", &special_energy,&special_energy_set); 
  if (special_energy_set) {
    targets.resize(0); targets.push_back(special_energy);
    myparameters.interval_set=PETSC_TRUE;
  }



  /******************************/
  // Defining sigmas
  //std::vector<Mat> sigmas;
  //sigmas.resize(L);
  std::vector<Vec> sigmas_as_vec;
  sigmas_as_vec.resize(L);
  for (int p=0;p<L;++p) { MatCreateVecs(H,  &sigmas_as_vec[p], NULL);}

  for (int k = 0; k < L; ++k) {
  VecAssemblyBegin(sigmas_as_vec[k]);
  VecAssemblyEnd(sigmas_as_vec[k]);
  }


  for (double &renorm_target : targets) {  // new setup
    double target = renorm_target * (Emaxc - Eminc) + Eminc;
    if (special_energy_set) { target=renorm_target;}
    EPS eps2;
    //		EPSInitializePackage();
    ierr = EPSCreate(PETSC_COMM_WORLD, &eps2);
    CHKERRQ(ierr);
    ierr = EPSSetOperators(eps2, H, NULL);
    CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps2, EPS_HEP);
    CHKERRQ(ierr);
    ST st;
    EPSGetST(eps2, &st);

    // Get and Set interval from my own parameters ...
    EPSSetFromOptions(eps2);

    std::string energy_name;
    char* eps_interval_string = new char[1000];
    PetscBool eps_interval_set=PETSC_FALSE;
    PetscOptionsGetString(NULL, NULL, "-eps_interval", eps_interval_string, 1000,&eps_interval_set); 
 
    if (special_energy_set) { eps_interval_set=PETSC_FALSE;}

    if ((!(eps_interval_set)) && (!(myparameters.interval_set))) { 
      EPSSetWhichEigenpairs(eps2, EPS_TARGET_REAL);
      EPSSetTarget(eps2, target);
      std::stringstream energy_string;
      energy_string << ".target=" << target;
      energy_name=energy_string.str();
    }

    if (eps_interval_set) {
      double Ea,Eb;
      if (myparameters.interval_set) {
      double born1 = myparameters.target1 * (Emaxc - Eminc) + Eminc;
      double born2 = myparameters.target2 * (Emaxc - Eminc) + Eminc;
      double epsilon=1e-2;
      if (myparameters.target1==0) { born1-=epsilon;}
      if (myparameters.target2==1) { born2+=epsilon;}
      EPSSetInterval(eps2, born1,born2);
      }

      std::stringstream energy_string;
      energy_string.precision(6);
      EPSGetInterval(eps2,&Ea,&Eb);
      double epsilona=(Ea-Eminc)/(Emaxc-Eminc);
      double epsilonb=(Eb-Eminc)/(Emaxc-Eminc);
      
      energy_string << ".targetinf=" << epsilona << ".targetsup=" << epsilonb;
      energy_name=energy_string.str();
    }

    ierr = EPSSolve(eps2);  

    PetscInt nconv = 0;
    ierr = EPSGetConverged(eps2, &nconv);
    CHKERRQ(ierr);
    if (0 == myrank) std::cout << "Solved done. \n";


    Vec use1;
    MatCreateVecs(H, &use1, NULL);
    std::vector<double> Aj_cum(L,0.);

    if (nconv > 0) {
      ofstream Aout;
      std::stringstream filename;
      filename << "A." << myparameters.string_from_basisstring_from_basis << myparameters.string_from_basisstring_from_H << ".dat";
      cout << filename.str() << endl;
      Aout.open((filename.str()).c_str());
      Aout.precision(20);

      for (int i = 0; i < nconv; i++) 
      {
        EPSGetEigenpair(eps2, i, &Er, &Ei, xr, NULL);
        std::vector<double> sz(L,0.);
        VecPointwiseMult(use1,xr,xr);
        for (int k=0;k<L;++k) { VecDot(use1,sigmas_as_vec[k],&sz[k]); }
        for (int k=0;k<L;++k) { Aj_cum[k]+=sz[k]*sz[k];}
      }
      for (int k=0;k<L;++k) { Aj_cum[k]/=nconv;}
      for (int k=0;k<L;++k) { Aout << Aj_cum[k]; if (k==(L-1)) { Aout << "\n"; } else { Aout << " ";} }
      Aout.close();

    } // nconv>0
  }// target
  SlepcFinalize();
  return 0;
}