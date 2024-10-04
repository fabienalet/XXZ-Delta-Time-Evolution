static char help[] = "Full ED for XXZ chain with Fibonacci fields \n";

//for cpu affinity check
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <unistd.h>

#define PETSC_DESIRE_COMPLEX
#define PETSC_USE_COMPLEX
#include <complex>
#include <slepceps.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <boost/dynamic_bitset.hpp>
#include<boost/random/mersenne_twister.hpp>
#include<boost/random/uniform_int.hpp>
#include<boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>


#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#else
typedef struct __MKL_Complex16 { double real; double imag; } MKL_Complex16;
typedef int MKL_INT;
typedef MKL_Complex16 lapack_complex_double;

#include <cblas.h>
#include <lapacke.h>
#endif

double PI=acos(-1.);
using namespace std;

#include "Spin/Spin_parameters.h"
#include "Spin/Spin_basis.h"
#include "Spin/SpinOneHalfXXZ.h"
#include "Spin/Spin_observable.h"
#include "Spin/ProductState.h"

#include "Geometry/Word.h"
#include "Geometry/SturmianSeq.h"

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc,char **argv)
{
  std::cout.precision(20);
  SlepcInitialize(&argc,&argv,"slepc.options",help);

  /**** Init parallel work ****/
  // For parallelization on node (openMP)
  int ENV_NUM_THREADS=mkl_get_max_threads(); /// Get value of OMP_NUM_THREADS
  omp_set_num_threads(ENV_NUM_THREADS); // Deactivate OpenMP for PETSc and SLEPc
  #ifdef USE_MKL
  mkl_set_num_threads(ENV_NUM_THREADS); // Deactivate OpenMP for PETSc and SLEPc
  #endif
  // For parallelization between nodes (MPI)
  int myrank, mpisize; MPI_Comm_rank(MPI_COMM_WORLD, &myrank); MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  //check CPU affinity
  char hostname[1024];
  gethostname(hostname, 1024);
  #pragma omp parallel
  {
    printf("On host %s, process %i (of %i), thread %i is on cpu %i\n", hostname, myrank, mpisize, omp_get_thread_num(), sched_getcpu());
  }

  /***** Petsc / Slepc data structure ***/
  Mat H;
  EPS eps; EPSType type;
  PetscErrorCode ierr;
  PetscReal error, tol, re, im;

  int number_of_states=2; double Sz=0;
  ierr = PetscOptionsGetReal(NULL,NULL,"-Sz",&Sz,NULL); CHKERRQ(ierr);

  Parameters myparameters(myrank);
  int L=myparameters.L;
  basis mybasis(L, Sz, myrank, number_of_states);
  int nconf=mybasis.total_number_of_confs;
  observable myobservable(&mybasis, ENV_NUM_THREADS);

  // generate a basis for each entanglement cut, and the associated observable objects
  std::vector<basis> bases;
  std::vector<observable> observables;
  std::vector<int> cuts;
  if (myparameters.num_times > 0 and myparameters.measure_entanglement_at_all_cuts)
  {
    for (int cut=3;cut<=L-3;cut++) cuts.push_back(cut);
    for (int cut: cuts)
    {
      basis cur_basis (L, cut, Sz, myrank, number_of_states);
      bases.push_back(cur_basis);
    }
    for (int idx=0;idx<bases.size();idx++)
    {
      observables.push_back(observable (&bases[idx], ENV_NUM_THREADS));
    }
  }

  /************************** Hamiltonian **********************************/
  double *H_1d;
  H_1d = (double*)calloc( ((long long int) nconf)*nconf,sizeof(double) );
  Hamiltonian myHamiltonian(&mybasis, &H);
  myHamiltonian.get_parameters();
  myparameters.string_from_basis=mybasis.string_names;
  myparameters.string_from_H=myHamiltonian.string_names;
  myHamiltonian.create_matrix_lapack(H_1d);
  // perform full ED
  double w[nconf];
  bool compute_eigenvectors = myparameters.eigenvectors;
  myHamiltonian.diagonalize_lapack(H_1d, w, compute_eigenvectors);

  /************************** Decoupled Hamiltonian **********************************/
  double* Hdec_1d;
  double w_dec[nconf];
  // The decoupled Hamiltonian needs to be diagonalize even is coupling time is set to 0 (this is not optimal!)
  if (myparameters.num_times > 0)
  {
    Hdec_1d = (double*)calloc( ((long long int) nconf)*nconf,sizeof(double) );
    Mat Hdec;
    Hamiltonian decHamiltonian(&mybasis, &Hdec);
    decHamiltonian.get_parameters();
    // set by hand a hopping element to 0, effectively decoupling left and right parts
    int decoupling_pos = L/2;
    decHamiltonian.coupling[decoupling_pos] = 0.;
    myHamiltonian.create_matrix_lapack(Hdec_1d);
    // perform full ED
    decHamiltonian.diagonalize_lapack(Hdec_1d, w_dec, true);
  }

  /*** Spectral measurements ***/
  // Initialize file names
  PetscReal Delta=1.;
  PetscReal h=0.;
  PetscInt shift=0;
  PetscBool pbc_value=PETSC_TRUE;
  PetscBool pbc=PETSC_TRUE;
  PetscReal Szmax=0;
  PetscReal Szmin=0;
  PetscReal slope = 0.6180339887498949;
  PetscBool palindrome_free = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL,"-pbc",&pbc_value,&pbc);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL,"-disorder",&h,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL,"-shift",&shift,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL,"-delta",&Delta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL,"-Szmin",&Szmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL,"-Szmax",&Szmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL,"-slope",&slope,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL,"-palindrome_free",&palindrome_free,NULL);CHKERRQ(ierr);

  std::stringstream boundary;
  if (pbc_value) boundary << ".pbc.shift=" << shift;
  else
  {
    boundary << ".open.shift=" << shift;
    if (palindrome_free) boundary << ".palindrome_free";
  }
  std::stringstream suffix;
  suffix << "L=" << L << ".SzTotal=" << Szmin <<  ".Delta=" << Delta << ".h=" << h << boundary.str() << ".dat";
  std::stringstream resfilename; resfilename << "rgap." << suffix.str();
  std::stringstream resfilename2; resfilename2 << "E." << suffix.str();
  std::ofstream rgapout((resfilename.str()).c_str()); rgapout.precision(20);
  std::ofstream enout((resfilename2.str()).c_str()); enout.precision(20);
  // Do measurements
  double gap, previous_gap, previous_E;
  for (int i=0;i<nconf;++i)
  {
  	// Energies and gap ratio
  	if (i!=0)
  	{
  		double gap=w[i]-previous_E;
  		if (i!=1) { double rg=gap/previous_gap; if (rg>1) { rg=1./rg;}
  		rgapout  << gap << " " << rg << std::endl; }
  		previous_gap=gap;
  	}
  	previous_E = w[i];
  	enout << previous_E << std::endl;
  }
  rgapout.close(); enout.close();

if (myparameters.eigenvectors)
{
  ofstream entout; ofstream locout; ofstream partout;
  myparameters.init_filenames_eigenstate_full(entout,locout,partout);

  std::ofstream correlout;
  std::ofstream correlPMout;
  if (myparameters.measure_correlations)
  {
    std::stringstream resfilenameCorrel; resfilenameCorrel << "Correl." << suffix.str();
    std::stringstream resfilenameCorrelPM; resfilenameCorrelPM << "CorrelPM." << suffix.str();
    correlout.open((resfilenameCorrel.str()).c_str()); correlout.precision(20);
    correlPMout.open((resfilenameCorrelPM.str()).c_str()); correlPMout.precision(20);
  }

  PetscScalar *cur_eigenstate;
  cur_eigenstate = (PetscScalar*)calloc( nconf,sizeof(PetscScalar) );
  for (int i=0;i<nconf;i++)
  {
    if (myrank==0)
    {
      for (int c=0;c<nconf;++c) cur_eigenstate[c] = H_1d[nconf*i + c];

      if (myparameters.measure_correlations)
      {
        // connected correlation
        myobservable.compute_two_points_correlation(cur_eigenstate);
        auto Gc = myobservable.get_two_points_connected_correlation(cur_eigenstate);
        correlout << "# Correlation matrix of the eigenstate " << i << " of energy " << w[i] << std::endl;
        for (auto &row: Gc)
        {
          for (double &correl: row)
          {
            correlout << correl << " ";
          }
          correlout << std::endl;
        }

        auto GPM = myobservable.get_SpSm_correlation(cur_eigenstate);
        correlPMout << "# PM correlation of the eigenstate " << i << " of energy " << w[i] << std::endl;
        for (auto &row: GPM)
        {
          for (double &correl: row)
          {
            correlPMout << correl << " ";
          }
          correlPMout << std::endl;
        }
      }

      if (myparameters.measure_all_part_entropy)
      {
        std::vector<double> entropies(myparameters.Nq+1,0.); entropies=myobservable.all_part_entropy(cur_eigenstate,myparameters.Nq,myparameters.qmin,myparameters.qmax);
        double qq=myparameters.qmin;
        for (int nq=0;nq<(myparameters.Nq+1);++nq) { partout << qq << " " << entropies[nq] << "\n"; qq+=(myparameters.qmax-myparameters.qmin)/(myparameters.Nq);}
      }

      else
      {
        double PE=myobservable.part_entropy(cur_eigenstate,1);
        //parout << PetscRealPart(Er) << " " << PE << "\n";}
        partout << PE << "\n";
      }

      if (myparameters.measure_entanglement)
      {
        myobservable.compute_entanglement_spectrum(cur_eigenstate);
        double S1=myobservable.entang_entropy(1);
        entout << S1 << "\n";
      }

      if (myparameters.measure_local)
      {
        myobservable.compute_local_magnetization(cur_eigenstate);
        std::vector<double> Siz=myobservable.sz_local;
        for (int r=0;r<L;++r) { locout << r << " " << Siz[r] << endl;}
      }
    }
  }
  free(cur_eigenstate);
  entout.close();locout.close(); partout.close();
  if (myparameters.measure_correlations)
  {
    correlout.close();
    correlPMout.close();
  }
}

  /**** Time evolution and related measurements *****/
  double tcoupling = myparameters.tcoupling;
  if (myparameters.num_times > 0 and myparameters.eigenvectors)
  {
    std::cout << "# Time evolution from with " << myparameters.num_times << " points on a linear time grid, with dt = " << myparameters.dt << "\n";
    std::cout << "Emin = " << w[0] << ", Emax = " << w[nconf-1] << endl;
    // Initialize time grid
    myparameters.Initialize_timegrid();

    // initialize vectors
    std::vector<double> result_vector_real (nconf);
    std::vector<double> result_vector_imag (nconf);
    std::vector<double> decoupled_vector_real (nconf);
    std::vector<double> decoupled_vector_imag (nconf);
    std::vector<double> result_vector2_real (nconf);
    std::vector<double> result_vector2_imag (nconf);

    // list of initial states (each of these consists of one basis vector)
    std::vector<int> init_states;
    int ineel = mybasis.ineel;
    if (myparameters.product_state_start)
    {
      // If no number of sample is specified, we start from just 1
      int seed3=15101976;
      ierr = PetscOptionsGetInt(NULL,NULL,"-seed3",&seed3,NULL);CHKERRQ(ierr);
      // a non-deterministic random number generator using the hardware
      //std::random_device rd;
      // generate a seed
      //seed3 = rd();
      // boost::mt19937 generator(seed3);
      int Nsamp = myparameters.num_product_states;
      double Emax = 0.125*L;
      double Emin = -Emax;
      // draw random configurations between Emin and Emax
      ProductState ps(0, &mybasis);
      init_states = ps.get_random_confs(Emin, Emax, Nsamp, seed3);
      // for (int i=0; i<Nstates;i++)
      // {
      //   boost::random::uniform_real_distribution<double> box(0, nconf);
      //   int i0 = box(generator);
      //   init_states.push_back(i0);
      // }
      if (myrank==0) {std::cout << "# " << Nsamp << " random product states initialized with seed3 = " << seed3 << std::endl;}
    }
    if (myparameters.cdw_start)
    {
      if(myrank==0)
      {
        std::cout << "# Starting from the Néel state = " << ineel << endl;
      }
      init_states.push_back(ineel);
    }
    else
    {
      if(myrank==0)
      {
        std::cout << "# Nothing specified. Starting from the Néel state = " << ineel << endl;
      }
      init_states.push_back(ineel);
    }

    // loop over initial states
    for (int i0: init_states)
    {
      // parameters of the initial configuration
      int nsa0, ca0, cb0;
      mybasis.get_conf_coefficients(i0, nsa0, ca0, cb0);
      // files
      ofstream entout; ofstream entcutout; ofstream imbout; ofstream retout; ofstream partout; ofstream locout;
      myparameters.init_filenames(entout, entcutout, imbout, locout, retout, partout, i0);

      // initialize state
      for (int k=0;k<nconf;++k) decoupled_vector_real[k] = 0.;
      decoupled_vector_real[i0] = 1.;

      /*
      Perform the decoupled time evolution (we do not record observables at this stage)
      At the end of this step, decoupled_vector_real and decoupled_vector_imag contain the
      vector at coupling time, written in the eigenstate basis
      */
      MKL_INT myn = nconf;
      MKL_INT incx=1; MKL_INT incy=1;
      char JOB='C'; // we will ask lapack to transpose the matrix
      double alpha, beta;  alpha=1.0; beta=0.0;
      dgemv(&JOB,&myn,&myn, &alpha,&Hdec_1d[0],&myn,&decoupled_vector_real[0], &incx, &beta, &result_vector2_real[0],&incy);
      double a,b,c,d;
      for (int k=0;k<nconf;++k)
      {
        a = cos(w_dec[k]*tcoupling);
        b = -sin(w_dec[k]*tcoupling);
        c = result_vector2_real[k];   // d=result_vector2_imag[k]; // no imag in this specific case
        decoupled_vector_real[k] = a*c; // a*c-b*d;
        decoupled_vector_imag[k] = b*c; // a*d+b*c;
      }
      // go back to real space (writing in result_vector2_*)
      JOB='N';
      dgemv(&JOB,&myn,&myn, &alpha,&Hdec_1d[0],&myn,&decoupled_vector_real[0], &incx, &beta, &result_vector2_real[0],&incy);
      dgemv(&JOB,&myn,&myn, &alpha,&Hdec_1d[0],&myn,&decoupled_vector_imag[0], &incx, &beta, &result_vector2_imag[0],&incy);
      // go to eigenbasis of coupled Hamiltonian (writing in decoupled_vector_*)
      JOB='C';
      dgemv(&JOB,&myn,&myn, &alpha,&H_1d[0],&myn,&result_vector2_real[0], &incx, &beta, &decoupled_vector_real[0],&incy);
      dgemv(&JOB,&myn,&myn, &alpha,&H_1d[0],&myn,&result_vector2_imag[0], &incx, &beta, &decoupled_vector_imag[0],&incy);

      // write cuts to file if any
      if (myparameters.measure_entanglement_at_all_cuts)
      {
        for (int cut: cuts)
        {
          entcutout << " " << cut;
        }
        entcutout << std::endl;
      }

      // Will be used for measurements
      PetscScalar * state;
      state = (PetscScalar*)calloc( nconf,sizeof(PetscScalar) );
      // Will be used for measurements in other bases
      PetscScalar * permuted_state;
      permuted_state = (PetscScalar*)calloc( nconf,sizeof(PetscScalar) );
      /****** Time loop ******/
      int t_index; double dt_measure=(myparameters.TEEmax-myparameters.TEEmin)/myparameters.nmeasures;
      double time_next_measure=myparameters.TEEmin;
      int each_measurement=myparameters.num_times/myparameters.nmeasures;
      for (t_index=0;t_index<=myparameters.num_times;++t_index)
      {
        double t=myparameters.time_points[t_index];
        // do time evolution of the initial eigenstate
        // (writing result in result_vector_real, result_vector_imag)
        double a,b,c,d;
        for (int k=0;k<nconf;++k)
        {
          a = cos(w[k]*t);
          b = -sin(w[k]*t);
          c = decoupled_vector_real[k];
          d = decoupled_vector_imag[k];
          // real part of \psi(t)
          result_vector_real[k] = a*c - b*d; // a*c-b*d;
          // imaginary part of \psi(t)
          result_vector_imag[k] = a*d + b*c; // a*d+b*c;
        }
        // write the time-evolved state (result_vector_real, result_vector_imag) in config space
        // (writing result in result_vector2_real, result_vector2_imag)
        JOB='N';
        MKL_INT incx=1; MKL_INT incy=1;
        double alpha, beta;  alpha=1.0; beta=0.0;
        dgemv(&JOB,&myn,&myn, &alpha,&H_1d[0],&myn,&result_vector_real[0], &incx, &beta, &result_vector2_real[0],&incy);
        dgemv(&JOB,&myn,&myn, &alpha,&H_1d[0],&myn,&result_vector_imag[0], &incx, &beta, &result_vector2_imag[0],&incy);

        if (myrank==0)	cout << "... Solved time t=" << t  << std::flush << std::endl;
        /************** Measurements ************/
        if ((t_index%each_measurement)==0)
        {
          for (int k=0;k<nconf;++k)
          {
            state[k] = result_vector2_real[k] + PETSC_i*result_vector2_imag[k];
          }

          if (myparameters.measure_local || myparameters.measure_imbalance)
          {
            myobservable.compute_local_magnetization(state);
            if (myparameters.measure_imbalance)
            {
              double Imb = myobservable.product_state_imbalance(nsa0, ca0, cb0);
              imbout << t << " " << Imb << endl;
              std::cout << "IMBALANCE " << i0 << " " << t << " " << Imb << std::endl;
            }
            if (myparameters.measure_local)
            {
              std::vector<double> Siz(L,0.);
              Siz = myobservable.sz_local;
              for (int r=0;r<L;++r){ locout << "SZ " << i0 << " " << t << " " << r << " " << Siz[r] << std::endl;}
            }
          }
          if (myparameters.measure_entanglement)
          {
            myobservable.compute_entanglement_spectrum(state);
            double S1 = myobservable.entang_entropy(1);
            entout << t << " " << S1 << endl;
            cout << "ENTANGLEMENT " << i0 << " " << t << " " << S1 << endl;
            if (myparameters.measure_entanglement_at_all_cuts)
            {
              entcutout << t;
              for (int idx=0;idx<observables.size();idx++)
              {
                observable obs = observables[idx];
                int cut = cuts[idx];
                // perform change of basis from mybasis to bases[idx]
                bases[idx].change_basis(mybasis, state, permuted_state);
                obs.compute_entanglement_spectrum(permuted_state);
                double S1 = obs.entang_entropy(1);
                entcutout << " " << S1;
                cout << "ENTANGLEMENT " << i0 << " " << t << " " << S1 << " AT CUT " << cut << endl;
              }
              entcutout << std::endl;
             }
          }
          if (myparameters.measure_return)
          {
            double ret = myobservable.return_probability(state,i0);
            retout << t << " " << ret << endl;
            cout << "RETURN " << i0 << " " << t << " " << ret << endl;
          }
          if (myparameters.measure_participation)
          {
            double P1 = myobservable.part_entropy(state,1);
            //double P1=0.;
            partout << t << " " << P1 << endl;
            cout << "PARTICIPATION " << i0 << " " << t << " " << P1 << endl;
          }
        } // end measurements
      } // end of t_index
      free( (void*)state );
      free(permuted_state);
      entout.close(); entcutout.close(); imbout.close(); retout.close(); partout.close();
    } // end of io loop
  }
  free( (void*)H_1d );
  return 0;
}
