static char help[] = "Time evolution for Google circuits\n"
                     "\n";

#define PETSC_DESIRE_COMPLEX
#include <complex>
#include <petscksp.h>

//#include <slepcmfn.h>
//#include <slepceps.h>
#include <random>
//#include <fstream>
#include <iostream>
//#include <map>
#include <omp.h>
//#include <slepceps.h>
//#include <sstream>
#include <malloc.h>
//#include<chrono>

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

using namespace std;

#include "Unitary_as_gates.h"

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **argv) {
  cout.precision(20);
  PetscInitialize(&argc,&argv,"floquet.options",help);

/**** Init parallel work ****/
  // For parallelization on node (openMP)
  
  int ENV_NUM_THREADS =omp_get_num_threads();
  int ENV_MAX_THREADS=omp_get_max_threads();
  omp_set_num_threads(1);
#ifdef USE_MKL
  ENV_MAX_THREADS = mkl_get_max_threads(); 
 // ENV_NUM_THREADS = mkl_get_max_threads(); 
  mkl_set_num_threads(1); // Deactivate OpenMP for PETSc and SLEPc
#endif

  // For parallelization between nodes (MPI)
  int myrank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  if (myrank==0) { std::cout << "ENV_MAX= " << ENV_MAX_THREADS << " ENV_NUM=" << ENV_NUM_THREADS << endl;}
  /***** Petsc / Slepc data structure ***/
  //Mat U;
  //EPS eps;
  //EPSType type;
  PetscErrorCode ierr;
  //PetscReal error, tol, re, im;
  //MFN mfn; FN fct;

  //Parameters myparameters(myrank);
    Unitary_as_gates *op = new Unitary_as_gates(myrank, mpisize);
  int L=op->Lchain_;

    //  malloc_trim(0);
 // cout << "Here 1\n";
  /*****  Get and initialize vectors *****/
	Vec Psi_t, res;// Vec Psi_t2,res2;
  MatCreateVecs(op->_U, NULL, &Psi_t); 
  MatCreateVecs(op->_U, NULL, &res); 
 // cout << "Here 2\n";
  VecAssemblyBegin(res); VecAssemblyEnd(res);
 // cout << "Here 3\n";
  /*****  Initialize vectors for sigma_z for fast measurements *****/
  PetscBool pbc=PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-pbc", &pbc, NULL);
  int L_for_sigma=L;
  if (pbc) {L_for_sigma=1;}
  Vec sigmas_as_vec; 
  MatCreateVecs(op->_U, NULL, &sigmas_as_vec);
  
  PetscScalar val=0.;
  for (int i=op->_Istart;i<op->_Iend;++i) {
    std::bitset<32> b(i);
    val=0.;
    for (int p=0;p<L_for_sigma;++p) { 
      if (b[p]) { val+=1;} else { val-=1.; }
      }
   VecSetValue(sigmas_as_vec,i,val,INSERT_VALUES);   
  }
  VecAssemblyBegin(sigmas_as_vec);   VecAssemblyEnd(sigmas_as_vec); 
  //cout << "Here 4\n";
   /***** Loop over initial states ****/

  // list of initial states (each of these consists of one basis vector)
  std::vector<unsigned long int> init_states;
  
  // If no number of sample is specified, we start from just 1111111111
  


  unsigned long int nconf=op->nconf;
  PetscInt num_product_states = 1;
  PetscBool num_product_states_set=PETSC_FALSE;
  PetscOptionsGetInt(NULL, NULL, "-num_product_states",&num_product_states, &num_product_states_set);  
  if (!(num_product_states_set)) {
  PetscOptionsGetInt(NULL, NULL, "-Nsamp",&num_product_states, &num_product_states_set);
  }
  if ((num_product_states_set)) {
  PetscInt seed3=15101976;
  PetscOptionsGetInt(NULL,NULL,"-seed3",&seed3,NULL);CHKERRQ(ierr);
  std::default_random_engine generator(seed3);
  std::uniform_int_distribution<> distribution(0, nconf-1);
  for (int r=0;r<num_product_states;++r) { init_states.push_back(distribution(generator));}
  if (myrank==0) {std::cout << "# " << num_product_states << " random product states initialized with seed3 = " << seed3 << std::endl;}
  }
  else {
    // special state 11111 has index nconf-1 ... TOCHECK
    init_states.resize(1); init_states[0]=nconf-1; //nconf-1;
    //std::cout << "HERE\n";
  }
  

  PetscBool measure_correlations=PETSC_FALSE;
  PetscBool measure_entanglement=PETSC_FALSE;
  PetscBool measure_return=PETSC_TRUE;
  PetscOptionsGetBool(NULL, NULL, "-measure_correlations", &measure_correlations, NULL);
  PetscOptionsGetBool(NULL, NULL, "-measure_entanglement", &measure_entanglement, NULL);
  PetscOptionsGetBool(NULL, NULL, "-measure_return", &measure_return, NULL);

  //  observable myobservable=observable(&mybasis,ENV_MAX_THREADS);

  Vec Vec_local;
	// loop over initial states
  
  for (auto i0: init_states)
  {
    VecSet(Psi_t,0.0);
    VecSetValue(Psi_t,i0,1.0,INSERT_VALUES);
    VecAssemblyBegin(Psi_t); VecAssemblyEnd(Psi_t);
   // VecAssemblyBegin(res); VecAssemblyEnd(res);
   
    //PetscReal norm;
    //VecNormalize(Psi_t,&norm);
    //VecCopy(Psi_t,res);
    //VecView(Psi_t,PETSC_VIEWER_STDOUT_WORLD); 
    /****** Time loop ******/
    PetscInt num_times=1;
    PetscOptionsGetInt(NULL, NULL, "-tmax", &num_times, NULL);  
    PetscOptionsGetInt(NULL, NULL, "-num_times", &num_times, NULL);
    PetscInt dt_measurement=10;
    PetscOptionsGetInt(NULL, NULL, "-dt", &dt_measurement, NULL);  
    PetscOptionsGetInt(NULL, NULL, "-dt_measurement", &dt_measurement, NULL);  
    PetscOptionsGetInt(NULL, NULL, "-measure_every", &dt_measurement, NULL);

    //if (special) { num_times=2;}

    
    int t=0;
    PetscBool print_basis=PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-print_basis", &print_basis, NULL);
    if (print_basis) {
    if (myrank==0) { std::cout << "Basis is : \n";
    for (int i=0;i<nconf;++i) {
    std::bitset<32> b(i);
    std::cout << i << " " << b << " ";
    for (int rr=0;rr<op->Lchain_;++rr) { cout << b[rr];}
    cout << endl;
    }
    }
    }
    
   // if (myrank==0) { cout << "Initial vector (time 0)\n";}
   // VecView(Psi_t,PETSC_VIEWER_STDOUT_WORLD);
    for (int t_index=0;t_index<=num_times;++t_index)
    {  
      if (myrank==0) std::cout << "Doing t=" << t+1 << endl;
       
        // do one time-evolution
      MatMult(op->_U,Psi_t,res);
      t++;
      
      //std::cout << "After evol t=" << t << endl;
      //VecView(res,PETSC_VIEWER_STDOUT_WORLD); 
      /************** Measurements ************/
      if ((t%dt_measurement)==0)
      {
        // Do direct measurement of sigma_z's
        // std::vector<double> sz(L_for_sigma,0.);
        double sz;
        // Careful I am using Psi_t as a temp vector to store res*res (point-wise)
        VecPointwiseMult(Psi_t,sigmas_as_vec,res);
        VecDotRealPart(Psi_t,res,&sz);  
        if (!(pbc)) { sz/=L;}
        if (myrank==0) { std::cout << "TIME " << t << " SZ " << sz << endl; }
        
        
        // Will rapatriate the distributed vector res into a local vector
        if (measure_entanglement) {
        Vec res_local;
        // Different strategies depending on MPI-distributed or not
        // only 1 mpi proc
        if (mpisize==1)
        {
          VecCreateSeq(PETSC_COMM_SELF,nconf,&res_local); ierr = VecCopy(res,res_local);
        }
        // more than 1 proc
        else
        {
          VecScatter ctx;
          VecScatterCreateToZero(res,&ctx,&res_local);
          VecScatterBegin(ctx,res,res_local,INSERT_VALUES,SCATTER_FORWARD); VecScatterEnd(ctx,res,res_local,INSERT_VALUES,SCATTER_FORWARD);
          VecScatterDestroy(&ctx);
        }

        // only the processor 0 will do the job ...
        if (myrank==0)
        {
          PetscScalar * state;
          // affect to whole res_local vector to processor 0
          VecGetArray(res_local, &state);

          if (measure_correlations)
          { }
          
          if (measure_return)
          { double a = PetscAbsScalar(state[i0]);
            double ret=a*a;
            cout << "RETURN " << i0 << " " << t << " " << ret << endl;
          }
          
          // Entanglemetn always at the end
          if (measure_entanglement)
          { 
            // move all possible basis
            //observable.compute_entanglement_spectrum3(newstate);
            //S1 = observable.entang_entropy(1);
            //cout << "ENTANGLEMENT " << i0 << " " << t << " " << (int) LAP << " " << S1 << endl;
          }
          // reaffect the data to each processor
          VecRestoreArray(res_local, &state);
        } // end of 0processor
        VecDestroy(&res_local);
      }
       
      } // end measurements

      /************** End of  Measurements ***********/
      // put back Res in Psi_t for next time evolution
      VecSwap(res, Psi_t);
    } // end of t_index
    
  } // end of io loop

  PetscFinalize();
  return 0;
}
