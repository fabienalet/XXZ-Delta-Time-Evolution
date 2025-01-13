static char help[] = "Krylov for XXZ chain \n (C) Fabien Alet 2024. \n\n";
/****** This program needs to be linked to a complex petsc-slepc installation !!! *********/

#define PETSC_DESIRE_COMPLEX
#include <complex>
#include <slepcmfn.h>
#include <slepceps.h>
#include <boost/dynamic_bitset.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <sstream>
#include <malloc.h>
#include <chrono>
#include <random>

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

//#include <cblas.h>
#include <lapacke.h>
#endif

double PI = acos(-1.);
using namespace std;

#include "Spin/Spin_parameters.h"

#include "Spin/Spin_basis.h"

#include "Spin/SpinOneHalfXXZ_disorder.h"

#include "Spin/Spin_observable.h"

#undef __FUNCT__
#define __FUNCT__ "main"


int main(int argc,char **argv)
{
  cout.precision(20);
  /************* Init Petsc and Slepc *********/
  SlepcInitialize(&argc,&argv,"krylov.options",help);

  /************* Init parallel work *********************************/
  // For parallelization on node (openMP)
  int ENV_NUM_THREADS=omp_get_num_threads();
  omp_set_num_threads(1);
  #ifdef USE_MKL
  ENV_NUM_THREADS=mkl_get_max_threads(); /// Get value of OMP_NUM_THREADS 
  mkl_set_num_threads(1);
  omp_set_num_threads(1);
  #endif
  // In case of openMP problems, one could desactivate openMP for the Krylov code
  // omp_set_num_threads(1);
  // #ifdef USE_MKL
  // mkl_set_num_threads(1);
  // #endif

  // For parallelization between nodes (MPI)
  int myrank, mpisize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  /******************* Init basis and observables ***************************/
  // spin 1/2 = 2 states per site
  int number_of_states = 2;

  // Sz=0 sector by default
  double Sz = 0;
  PetscOptionsGetReal(NULL, NULL, "-Sz", &Sz, NULL);

  // Parameters taken from the command line
  Parameters myparameters(myrank);
  // Get chains size L from parameters
  int L=myparameters.L;

  // init basis
  basis mybasis(L, Sz, myrank, number_of_states);
  int nconf = mybasis.total_number_of_confs;
  if (myrank == 0) {
    std::cout << "# L = " << L << " number of states=" << nconf << std::endl;
  }
  // init observable
  observable myobservable(&mybasis, ENV_NUM_THREADS);

  // Petsc data structure for the matrix
  PetscInt Istart, Iend;
  Mat H;

  int ineel=mybasis.ineel;
  // store fields for computing average energy later
  std::vector<double> field (L);


  /************************** Hamiltonian **********************************/
  {
    // distribute evenly the matrix lines between processes
    std::vector<int> local_block_sizes(mpisize,nconf/mpisize);
    for(size_t i=0; i< nconf%mpisize; i++) local_block_sizes[i]++;
    Istart=0; for(size_t i=0; i<myrank; i++) Istart+=local_block_sizes[i];
    Iend=Istart+local_block_sizes[myrank];

    // init Hamiltonian
    Hamiltonian myHamiltonian(&mybasis,&H);
    myHamiltonian.get_parameters();
    myparameters.string_from_basis = mybasis.string_names;
    myparameters.string_from_H = myHamiltonian.string_names;
    field = myHamiltonian.field;

    myHamiltonian.init_matrix_sizes(Istart,Iend);
    MatSetUp(H);
    myHamiltonian.create_matrix(Istart,Iend);

  }

  /****  Matrix assembly ****/
  MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE);
  MatSetOption(H,MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
  MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);
  if(myrank==0) std::cout << "#Hamiltonian matrix assembly done." << std::endl;


  /************** Compute the edge energies of the spectrum ****************/
  // SLEPc data structure for the extremal eigenvalue problem
  EPS eps;
  PetscScalar Eminc,Emaxc;
  double Emin,Emax;
  bool do_ev=1;
  if (do_ev)
  {
    EPSCreate(PETSC_COMM_WORLD,&eps);
    EPSSetOperators(eps,H,NULL);
    EPSSetProblemType(eps,EPS_HEP);
    EPSSetDimensions(eps,1,PETSC_DECIDE,PETSC_DECIDE);

    // get the ground-state energy (in this sector)
    EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);
    EPSSolve(eps);
    EPSGetEigenvalue(eps,0,&Eminc,NULL);
    Emin=PetscRealPart(Eminc);
    if(0==myrank) std::cout << "Emin of H = " << Emin << "\n";

    // get the maximal energy (in this sector)
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
    EPSSolve(eps);
    EPSGetEigenvalue(eps, 0, &Emaxc, NULL);
    Emax = PetscRealPart(Emaxc);
    if (0 == myrank) std::cout << "Emax of H = " << Emax << "\n";
  }


  /******************************/
  // Defining sigmas
  //std::vector<Mat> sigmas;
  //sigmas.resize(L);
  std::vector<Vec> sigmas_as_vec;
  sigmas_as_vec.resize(L);
  for (int p=0;p<L;++p) { MatCreateVecs(H, NULL, &sigmas_as_vec[p]);}
  std::vector<Vec> sigmasigma_as_vec;
  // here give the numbers of pairs
  // int nb_pairs=(L*L/2);
  int nb_pairs=(L/2);
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
                if (config[k]) { VecSetValue(sigmas_as_vec[k],row_ctr,1.,INSERT_VALUES); }
                else { VecSetValue(sigmas_as_vec[k],row_ctr,-1.,INSERT_VALUES); }
          }
        }
        row_ctr++;
      }
    }
  }


  for (int k = 0; k < L; ++k) {
  VecAssemblyBegin(sigmas_as_vec[k]);
  VecAssemblyEnd(sigmas_as_vec[k]);
  }

  int running_pair=0;
//  for (int k=0;k<L;++k) {
  for (int k=0;k<(L/2);++k) { // for L/2 distance only, only need the first L/2 spins
    for (int range=(L/2);range<=(L/2);++range) { // only measure at L/2
      // for (int range=1;range<=(L/2);++range) { 
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


  /************************* Krylov computation *************************/
	// if(myrank==0)
  // {
  //     std::cout << "# nconf= " << nconf << std::endl;
  //     std::cout << "#Time evolution with " << myparameters.num_times << " points\n";
  // }

  /**** Initialize Krylov from Slepc *****/
  MatScale(H,-1.0*PETSC_i);
  MFN mfn;
  FN fct;
  MFNCreate(PETSC_COMM_WORLD, &mfn);
  MFNSetOperator(mfn, H);
  MFNGetFN(mfn, &fct);
  FNSetType(fct, FNEXP);
  MFNSetTolerances(mfn, 1e-12, PETSC_DEFAULT);
  MFNSetFromOptions(mfn);

  /*****  Get and initialize vectors *****/
  Vec Psi_t, res;
  MatCreateVecs(H, NULL, &Psi_t);
  MatCreateVecs(H, NULL, &res);

  /********* Initialize time grid ********/
  myparameters.Initialize_timegrid();
  /********* Initial state ***************/

  /***** Loop over initial states ****/

  // list of initial states (each of these consists of one basis vector)
 


  std::vector<unsigned long int> init_states;
  if (myparameters.product_state_start)
  { // Here we find initial basis states which have an energy within deps of the central energy
    // First we get the diagonal of the Hamiltonian to compute this energy
    // This is stored in the diag_local local vector as only processor 0 will do the job 
    Vec Diagonal;
    MatCreateVecs(H, NULL, &Diagonal);
    VecAssemblyBegin(Diagonal); VecAssemblyEnd(Diagonal);
    MatGetDiagonal(H,Diagonal);
    VecScatter ctx;
    Vec diag_local;
    if (mpisize!=1) {
    VecScatterCreateToZero(Diagonal, &ctx, &diag_local);
    VecScatterBegin(ctx, Diagonal, diag_local, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, Diagonal, diag_local, INSERT_VALUES, SCATTER_FORWARD);
    }
    else { VecCreateSeq(PETSC_COMM_SELF,nconf,&diag_local); VecCopy(Diagonal,diag_local);  }
    // If no number of sample is specified, we start from just 1
    int seed3=15101976;
    PetscOptionsGetInt(NULL,NULL,"-seed3",&seed3,NULL);
    int Nsamp = myparameters.num_product_states;
    double epsmin=0.45;
    double epsmax=0.55;
    PetscOptionsGetReal(NULL,NULL,"-epsmin",&epsmin,NULL);
    PetscOptionsGetReal(NULL,NULL,"-epsmax",&epsmax,NULL);
    double deps=fabs(epsmax-epsmin);
    PetscOptionsGetReal(NULL,NULL,"-deps",&deps,NULL);
    deps=fabs(deps);
    epsmin=0.5-deps/2;
    epsmax=0.5+deps/2;

    // only 0 will do the job
    if (myrank==0) {
      int Nsamp = myparameters.num_product_states;
      std::random_device device;
      std::mt19937 generator(device());
      generator.seed(seed3);
      std::uniform_int_distribution<> distr(0, nconf - 1);
      unsigned long int start=distr(generator);
      PetscScalar value; unsigned long int tries=0; int nfound=0;
      while((nfound<Nsamp) && (tries<nconf))
        { start=(start+1)%nconf;
          int istart=(int) start;
          VecGetValues(diag_local, 1,  &istart, &value);
          double this_deps=(-PetscImaginaryPart(value)-Emin)/(Emax-Emin);
          if (this_deps>=epsmin && this_deps<=epsmax) { 
            // do nothing
          }
          else {
            init_states.push_back(start); nfound++;
          }
        tries++;
        }
    std::cout << "# " << Nsamp
                << " random product states initialized with seed_inistates = "
                << seed3 << std::endl;
    }
    VecScatterDestroy(&ctx);
    VecDestroy(&diag_local);
  }
  else if (myparameters.cdw_start)
  {
    if(myrank==0)
    {
      std::cout << "# Starting from Neel state = " << ineel << endl;
    }
    init_states.push_back(ineel);
  }
  else
  {

    if (myparameters.special_state_start)
    { unsigned long int ii=mybasis.index(myparameters.special_conf);
      if(myrank==0) { cout << "#Starting with Special state = "; for (int r=0;r<L;r++) { std::cout << myparameters.special_conf[r];}
      cout << " with index= " << ii << endl;}
      init_states.push_back(ii);}
    else {
    if(myrank==0)
    {
      std::cout << "# Nothing specified. Starting from Néel state = " << ineel << endl;
    }
    init_states.push_back(ineel);
    }
  }

  // Need to broadcast init_state, in 2 steps
  int this_size=init_states.size();
  MPI_Bcast(&this_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (myrank!=0) { init_states.resize(this_size);}
  MPI_Bcast(&init_states[0], this_size, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if(myrank==0)
  {
    std::cout << "Using " << init_states.size() << " initial states ";
    for (auto i0: init_states) std::cout << i0 << " ";
    std::cout << std::endl;
  }


  Vec Vec_local;
	ofstream corrout; ofstream partout;ofstream entout; ofstream retout; ofstream imbout; ofstream locout;
  PetscBool omp_switch=PETSC_FALSE;
  std::vector<double> Siz(L, 0.); 
  std::vector< std::vector<double> > Gij(L); for (int k=0;k<L;++k) { Gij[k].resize(L,0.);}

/*********** loop over initial states **********/
  for (auto i0: init_states)
  { // parameters of the initial configuration
    

    int nsa0, ca0, cb0;
    mybasis.get_conf_coefficients(i0, nsa0, ca0, cb0);
    // creating the filenames (could be improved/modified)
    if(myrank==0){
    if (myparameters.measure_correlations) { myparameters.init_filename_correlations(corrout,i0);  PetscOptionsGetBool(NULL, NULL, "-omp", &omp_switch, NULL);   }
    if (myparameters.measure_participation) { myparameters.init_filename_participation(partout,i0);	}
    if (myparameters.measure_local) { myparameters.init_filename_imbalance(locout,i0);	}
    if (myparameters.measure_imbalance) { myparameters.init_filename_imbalance(imbout,i0);	    }
    if (myparameters.measure_entanglement) {  myparameters.init_filename_entanglement(entout,i0); entout.precision(20);
      //  PetscOptionsGetInt(NULL, NULL, "-LAmin", &LAmin, NULL); PetscOptionsGetInt(NULL, NULL, "-LAmax", &LAmax, NULL);
       }
      if (myparameters.measure_return) { myparameters.init_filename_return(retout,i0); }
      }

    // Initialise Psi(0) (with the initial state) and Res(0)
    VecSet(Psi_t,0.0);
    VecSetValue(Psi_t,i0,1.0,INSERT_VALUES);
    VecAssemblyBegin(Psi_t); VecAssemblyEnd(Psi_t);
    VecAssemblyBegin(res); VecAssemblyEnd(res);

    // Just to make sure: compute the norm
    PetscReal norm;
    VecNormalize(Psi_t,&norm);
    VecCopy(Psi_t,res);

    PetscBool debug=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-debug", &debug, NULL); 

    PetscBool print_entanglement_spectrum=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-print_entanglement_spectrum", &print_entanglement_spectrum, NULL); 

    /****** Time loop ******/
    int t_index;
    double dt_measure=(myparameters.TEEmax-myparameters.TEEmin)/myparameters.nmeasures;
    double time_next_measure=myparameters.TEEmin;
    int each_measurement=myparameters.num_times/myparameters.nmeasures;
    for (t_index=0;t_index<=myparameters.num_times;++t_index)
    {
      double t=myparameters.time_points[t_index];
      if (myparameters.delta_t_points[t_index] != 0) {
      FNSetScale(fct, (PetscScalar) myparameters.delta_t_points[t_index], (PetscScalar) 1.0);
      // result of time evolution written in res
      MFNSolve(mfn, Psi_t, res);
      if (myrank==0)	std::cout << "... Solved time t=" << t  <<  " " << myparameters.delta_t_points[t_index] << std::flush << std::endl;
      }
      /************** Measurements ************/
      if ((t_index%each_measurement)==0)
      {
        if ( (1) || (myparameters.measure_correlations) ) { // Measure sz-sz connected correlations in parallel (always for this code)
        // Careful I am using Psi_t as a temp vector to store res*res (point-wise)
        if (myrank==0) { cout << "*** Wathcing res \n";}
        VecView(res, PETSC_VIEWER_STDOUT_WORLD);
        VecPointwiseMult(Psi_t,res,res);
        if (myrank==0) { cout << "*** Wathcing res*res \n";}
        VecView(Psi_t, PETSC_VIEWER_STDOUT_WORLD);
        if (myrank==0) { cout << "*** Wathcing sz[0] \n";}
        VecView(sz[0], PETSC_VIEWER_STDOUT_WORLD);
        // First measure sz
        std::vector<double> sz(L,0.);
        for (int k=0;k<L;++k) { VecDotRealPart(Psi_t,sigmas_as_vec[k],&sz[k]); }
        if (myrank==0) { cout << "*** Wathcing dot-real-part= " << sz[0] << " \n";}
        if (myparameters.measure_local) { 
          if (myrank==0) {
          for (int k=0;k<L;++k) { locout << "SZ " << i0 << " " << t << " " << k+1 << " " << 0.5*sz[k] << std::endl;}
          }
        }

        // Then measure sz-sz correl
        double C=0.; int running_pair=0;
        for (int k=0;k<(L/2);++k) { // for L/2 distance only, only need the first L/2 spins
          for (int range=(L/2);range<=(L/2);++range) { // only measure at L/2
              VecDotRealPart(sigmasigma_as_vec[running_pair],Psi_t,&C);
              C=0.25*(C-sz[k]*sz[(k+range)%L]);
              running_pair++;
              if (myrank==0) { corrout << t << " " << k+1 << " " << k+range+2 << " " << C  << endl; }
        }
        }
        }
        // Will repatriate the distributed vector res into a local vector
        Vec res_local;
        // Different strategies depending on MPI-distributed or not
        // only 1 mpi proc
        if (mpisize==1)
        {
          VecCreateSeq(PETSC_COMM_SELF,nconf,&res_local);
          VecCopy(res,res_local);
        }
        // more than 1 proc
        else
        {
          VecScatter ctx;
          VecScatterCreateToZero(res,&ctx,&res_local);
          VecScatterBegin(ctx,res,res_local,INSERT_VALUES,SCATTER_FORWARD);
          VecScatterEnd(ctx,res,res_local,INSERT_VALUES,SCATTER_FORWARD);
          VecScatterDestroy(&ctx);
        }


        // only the processor 0 will do the job ...
        if (myrank==0)
        { 
          PetscScalar *state;
          // provide the whole res_local vector to processor 0
          VecGetArray(res_local, &state);
          /*
          if (myparameters.measure_correlations) {
            Gij =myobservable.get_two_points_connected_correlation(state);
            for (int r = 0; r < L; ++r) {
              for (int s = r; s < L; ++s) {
	              corrout << r << " " << " " << s << " " << Gij[r][s] << endl;
            } }
          }
          */
          /*
          if (myparameters.measure_transverse_correlations){
            // get two points transverse correlations
            Tij = myobservable.get_SpSm_correlation(state);
            for (int r = 0; r < L; ++r) {
              for (int s = r; s < L; ++s) {
                tcorrout << r << " " << " " << s << " " << Tij[r][s] << endl;
            } }
          }
          */
        if (myparameters.measure_imbalance)
            { // TODO CHECK IF IMPROVABLE
            if (!(myparameters.measure_local)) { myobservable.compute_local_magnetization(state);
            Siz = myobservable.sz_local; }
              double Imb = myobservable.product_state_imbalance(nsa0, ca0, cb0);
              imbout << t << " " << Imb << endl;
              std::cout << "IMBALANCE " << i0 << " " << t << " " << Imb << std::endl;
            }
                      if (myparameters.measure_return)
          { double a = PetscAbsScalar(state[i0]);
            double ret=a*a;
            retout << t << " " << ret << endl;
          }

          if (myparameters.measure_participation)
          {
            double P1=myobservable.part_entropy(state,1);
            partout << t << " " << P1 << endl;
            //cout << "PARTICIPATION " << i0 << " " << t << " " << P1 << endl;
          }
          if (myparameters.measure_entanglement)
          { if (debug) { myobservable.compute_entanglement_spectrum_debug(state); }
              else { myobservable.compute_entanglement_spectrum(state); }
            if (print_entanglement_spectrum) {
              for (int pp=0;pp<myobservable.entanglement_spectrum.size();++pp) {
                cout << "ES " << pp << " " << myobservable.entanglement_spectrum[pp] << endl;
              }
            }
            double S1=myobservable.entang_entropy(1);
            entout << t << " " << S1 << endl;
            cout << "ENTANGLEMENT " << i0 << " " << t << " " << S1 << endl;
          }

          // reaffect the data to each processor
          VecRestoreArray(res_local, &state);
        } // end of 0processor
        VecDestroy(&res_local);
      } // end measurements

      /************** End of  Measurements ***********/
      // put back Res in Psi_t for next time evolution
      VecCopy(res, Psi_t);
    } // end of t_index
    if (myrank==0) {  entout.close(); imbout.close(); retout.close(); partout.close(); corrout.close(); }
  } // end of io loop over initial states

  SlepcFinalize();
  return 0;
}
