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
int ENV_NUM_THREADS=omp_get_num_threads();
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
  MatCreateVecs(H, NULL, &xr);
  Vec xr2;
  MatCreateVecs(H, NULL, &xr2);
  
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


  /******************************/
  // Defining sigmas
  //std::vector<Mat> sigmas;
  //sigmas.resize(L);
  std::vector<Vec> sigmas_as_vec;
  sigmas_as_vec.resize(L);
  for (int p=0;p<L;++p) { MatCreateVecs(H, NULL, &sigmas_as_vec[p]);}
  std::vector<Vec> sigmasigma_as_vec;
  int nb_pairs=(L*L/2);
  sigmasigma_as_vec.resize(nb_pairs);
  for (int p=0;p<nb_pairs;++p) { MatCreateVecs(H, NULL, &sigmasigma_as_vec[p]);}

  //for (int k=0;k<L;++k) { MatCreateVecs(H, NULL, &sigmas[k]);}
  {
  //std::vector<PetscInt> d_nnz(Iend - Istart,1);
  //std::vector<PetscInt> o_nnz(Iend - Istart,0);
 // for (int k=0;k<L;++k) { MatCreateAIJ(PETSC_COMM_WORLD, Iend - Istart, PETSC_DECIDE,nconf,nconf, 0, d_nnz.data(), 0,
  //             o_nnz.data(), &(sigmas[k])); } // check &(sigmas[k]
  }
  int row_ctr = 0;
  for (int nsa = 0; nsa < mybasis.valid_sectors; ++nsa) {
    for (int ca = 0; ca < mybasis.Confs_in_A[nsa].size(); ++ca) {
      std::vector<unsigned short int> config(L, 0);
      std::vector<unsigned short int> confA =mybasis.Confs_in_A[nsa][ca];
      for (int r = 0; r < mybasis.LA; ++r) {config[r] = confA[r];}
      for (int cb = 0; cb < mybasis.Confs_in_B[nsa].size(); ++cb) {
        if ((row_ctr >= Istart) && (row_ctr < Iend)) {
          std::vector<unsigned short int> confB =mybasis. Confs_in_B[nsa][cb];
          for (int r = 0; r < mybasis.LB; ++r) { config[r + mybasis.LA] = confB[r]; }
          for (int k = 0; k < L; ++k) {
                if (config[k]) { 
 //                 MatSetValue(sigmas[k], row_ctr, row_ctr, (PetscScalar)1., ADD_VALUES);
                  VecSetValue(sigmas_as_vec[k],row_ctr,1.,INSERT_VALUES);
                  }
                else { 
   //               MatSetValue(sigmas[k], row_ctr, row_ctr, (PetscScalar)-1., ADD_VALUES);
                  VecSetValue(sigmas_as_vec[k],row_ctr,-1.,INSERT_VALUES);
                  }
          }
        }
        row_ctr++;
      }
    }
  }


  for (int k = 0; k < L; ++k) {
  //MatSetOption(sigmas[k], MAT_SYMMETRIC, PETSC_TRUE);
  //MatSetOption(sigmas[k], MAT_SYMMETRY_ETERNAL, PETSC_TRUE);
  //ierr = MatAssemblyBegin(sigmas[k], MAT_FINAL_ASSEMBLY);
  //ierr = MatAssemblyEnd(sigmas[k], MAT_FINAL_ASSEMBLY);
  VecAssemblyBegin(sigmas_as_vec[k]);
  VecAssemblyEnd(sigmas_as_vec[k]);
  }

  int running_pair=0;
  for (int k=0;k<L;++k) {
    for (int range=1;range<=(L/2);++range) { 
      VecPointwiseMult(sigmasigma_as_vec[running_pair],sigmas_as_vec[k],sigmas_as_vec[(k+range)%L]);
   //   if (1) { if (myrank==0) { cout << "running_pair=" << running_pair << " sites " << k << " " << (k+range)%L << endl;} }
      running_pair++;
    }
  }
  for (int p=0;p<nb_pairs;++p)
  {
  VecAssemblyBegin(sigmasigma_as_vec[p]);
  VecAssemblyEnd(sigmasigma_as_vec[p]);
  }
  /******************************/

  for (double &renorm_target : targets) {  // new setup
    double target = renorm_target * (Emaxc - Eminc) + Eminc;
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
    MatCreateVecs(H, NULL, &use1);

      ofstream entout;
      ofstream locout;
      ofstream alllocout;
      ofstream partout;
      ofstream corrout;
      ofstream tcorrout;
      ofstream KLout;
      ofstream sigmaout;

      if (myparameters.measure_KL) { myparameters.init_filename_KL(KLout,energy_name);}
      if (myparameters.measure_local) { myparameters.init_filename_local(locout,energy_name);       }
      if (myparameters.measure_correlations) { myparameters.init_filename_correlations(corrout,energy_name);}
      if (myparameters.measure_transverse_correlations) { myparameters.init_filename_transverse_correlations(tcorrout,energy_name);}
      if (myparameters.measure_participation) { myparameters.init_filename_participation(partout,energy_name);}
      if (myparameters.measure_entanglement) { myparameters.init_filename_entanglement(entout,energy_name);}

  
      std::vector<double> energies;
      std::vector<double> rgap;
      PetscScalar Er, Ei;
      Vec Vec_local;
      std::vector<double> Siz(L, 0.);
      std::vector< std::vector<double> > Gij(L);
      for (int k=0;k<L;++k) { Gij[k].resize(L,0.);}
      std::vector< std::vector<double> > Tij(L);
      for (int k=0;k<L;++k) { Tij[k].resize(L,0.);}

      PetscBool debug=PETSC_FALSE;
      PetscOptionsGetBool(NULL, NULL, "-debug", &debug,NULL);

      PetscBool measure_sigma_indicator=PETSC_TRUE;
      PetscOptionsGetBool(NULL, NULL, "-measure_sigma", &measure_sigma_indicator,NULL); 
      if (measure_sigma_indicator) { myparameters.init_filename_sigma(sigmaout,energy_name);}

      PetscBool other_measurements=PETSC_FALSE;
      PetscOptionsGetBool(NULL, NULL, "-other_measurements", &other_measurements,NULL); 


      // TODO : get pbc back !
      PetscBool pbc=PETSC_TRUE;
      PetscOptionsGetBool(NULL, NULL, "-pbc", &pbc, NULL);

      PetscReal KL_cutoff=2.0;
      PetscOptionsGetReal(NULL, NULL, "-KL_cutoff", &KL_cutoff,NULL); 

     // std::vector<bool> observables_computed_for_eigenstate(nconv,0);
      for (int i = 0; i < nconv; i++) 
      {
        EPSGetEigenpair(eps2, i, &Er, &Ei, xr, NULL);
        energies.push_back(PetscRealPart(Er));

        std::vector<int> j_paired; j_paired.resize(0);
        double minKL=1000000; int j_minKL=0; std::vector<std::pair<double,double> > saved_KLs;
        for (int j = i+1; j < nconv; j++) 
          { //if (i!=j) {
            double Er2;
            EPSGetEigenpair(eps2, j, &Er2, &Ei, xr2, NULL);
            double pi; double qi; double local_KL=0.; double local_KL2=0.;
            for (int row_ctr = Istart; row_ctr<Iend;++row_ctr) {
              VecGetValues( xr, 1, &row_ctr, &pi );
              VecGetValues( xr2, 1, &row_ctr, &qi );
                  if ((pi != 0) && (qi != 0)) {
                      local_KL += 2.0 * pi * pi * log(fabs(pi / qi));
                      local_KL2 += 2.0 * qi * qi * log(fabs(qi / pi));
                      }
              }
            double global_KL;
            double global_KL2;
            MPI_Reduce(&local_KL, &global_KL, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);
            MPI_Reduce(&local_KL2, &global_KL2, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);

            if (global_KL<minKL) { minKL=global_KL; j_minKL=j;}
            if (global_KL2<minKL) { minKL=global_KL2; j_minKL=j;}
            if ((global_KL<KL_cutoff) || (global_KL2<KL_cutoff)) { j_paired.push_back(j); saved_KLs.push_back(std::make_pair(global_KL,global_KL2));}
          cout << i << " " << j << " " << global_KL << " " << global_KL2 << endl;
            //}
          }


      for (int jp=0;jp<j_paired.size();++jp) {
        double Er2;
        cout << "Entering and searching for " << j_paired[jp] << endl;
        EPSGetEigenpair(eps2, j_paired[jp], &Er2, &Ei, xr2, NULL);
        cout << "Got " << j_paired[jp] << endl;
        if (myrank==0) { KLout << saved_KLs[jp].first << " " << saved_KLs[jp].second << " " << Er << " " <<  Er2 << " " << i << " " << j_paired[jp];
        if (j_paired[jp]==j_minKL) { KLout << " 1" << endl; } else { KLout << " 0" << endl;}
          }
        
        if (measure_sigma_indicator) {
              // TODO Optimize
              std::vector<double> sigma_indicator(L,0);
              VecPointwiseMult(use1,xr2,xr);
              for (int k=0;k<L;++k) { VecDot(use1,sigmas_as_vec[k],&sigma_indicator[k]); }
              if (myrank==0) {
            for (int k=0;k<L;++k) { sigmaout << "Sig " <<  k << " " << sigma_indicator[k] << " " << Er << " " <<  Er2 << " " << i << " " << j_paired[jp];
            if (j_paired[jp]==j_minKL) { sigmaout << " 1" << endl; } else { sigmaout << " 0" << endl;}
              }
            } 
        }

      }
      if (j_paired.size()>0) {
        //observables_computed_for_eigenstate[i]=1;
        std::vector<double> sz(L,0.);
          VecPointwiseMult(xr2,xr,xr);
          for (int k=0;k<L;++k) {
          VecDot(xr2,sigmas_as_vec[k],&sz[k]);
          }
          if (myparameters.measure_local) { 
            for (int k=0;k<L;++k) { locout << k+1 << " " << 0.5*sz[k] << " " << Er << " " << i << endl; } 
            }      
          if (myparameters.measure_correlations) {
            double C;
           for (int k=0;k<L;++k) 
                { for (int range=1;range<=(L/2);++range) 
                  { VecDot(sigmasigma_as_vec[running_pair],xr2,&C);
                    if (myrank==0) { corrout << k+1 << " " << (k+range)%L+1 << " " << 0.25*(C-sz[k]*sz[(k+range)%L]) << " " << Er << " " << i << endl; }    
                  }
                }
          }
          if (myparameters.measure_participation) {
          double pi; double local_S1=0.;
          for (int row_ctr = Istart; row_ctr<Iend;++row_ctr) {
            VecGetValues( xr, 1, &row_ctr, &pi );
             if ((pi != 0)) { local_S1 -= 2.0* pi * pi * log(fabs(pi));}
          }
          double global_S1=0.;
          MPI_Reduce(&local_S1, &global_S1, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);
          if (myrank==0) { partout << "S1 " << global_S1 << " " << Er << " " << i << endl; }
        }

     // other measurements on eigenstate to follow ...
    PetscBool other_measurements=PETSC_FALSE;
    if ((myparameters.measure_transverse_correlations) || (myparameters.measure_entanglement)) 
      { other_measurements=PETSC_TRUE;}
    if (other_measurements) {
        if (mpisize == 1) {
          VecCreateSeq(PETSC_COMM_SELF, nconf, &Vec_local);
          ierr = VecCopy(xr, Vec_local);
        }  // only 1 mpi proc
        else {
          VecScatter ctx;
          VecScatterCreateToZero(xr, &ctx, &Vec_local);
          VecScatterBegin(ctx, xr, Vec_local, INSERT_VALUES, SCATTER_FORWARD);
          VecScatterEnd(ctx, xr, Vec_local, INSERT_VALUES, SCATTER_FORWARD);
          VecScatterDestroy(&ctx);
        }
       
        if (myrank == 0) {
          PetscScalar *state;
          VecGetArray(Vec_local, &state);
          if (myparameters.measure_transverse_correlations){
            // get two points transverse correlations
            Tij = myobservable.get_SpSm_correlation(state);
            for (int r = 0; r < L; ++r) {
              for (int s = r; s < L; ++s) {
                tcorrout << r+1 << " " << " " << s+1 << " " << Tij[r][s] << " " << Er << endl;
            } }
          }
          if (myparameters.measure_entanglement) 
          {
            if (debug) { myobservable.compute_entanglement_spectrum_debug(state);}
            else { myobservable.compute_entanglement_spectrum(state);}
            if (myparameters.measure_entanglement_spectrum) { 
              std::vector<double> es=myobservable.entanglement_spectrum;
              std::sort(es.begin(), es.end());
              cout << "*** ES " << Er << endl;
              for (int p=0;p<myobservable.entanglement_spectrum.size();++p) {
                cout << es[p] << endl;
              }
            }
            double S1 = myobservable.entang_entropy(1);
            entout << S1 << "\n";
           // double S2=myobservable.entang_entropy(2);
           // cout << "S2 = " << S2 << " " << Er << std::endl;
          }
          VecRestoreArray(Vec_local, &state);
        }
        VecDestroy(&Vec_local);
      }
      
      }


      } // loop over eigenstates
    if (myparameters.measure_KL) {KLout.close();}
    if (myparameters.measure_local) { locout.close();  }
    if (myparameters.measure_correlations) { corrout.close();}
    if (myparameters.measure_participation) { partout.close();}
    if (myparameters.measure_entanglement) { entout.close();}
    if (measure_sigma_indicator) {sigmaout.close();}
    if (myparameters.measure_transverse_correlations){ tcorrout.close();}


  }// target
  SlepcFinalize();
  return 0;
}