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
  PetscBool output_matrix=PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL,"-output_matrix",&output_matrix,NULL);

    if (output_matrix) {
      PetscViewer    view;
      PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Matrix.mtx",&view);
      PetscViewerPushFormat(view,PETSC_VIEWER_ASCII_MATRIXMARKET);
      MatView(H,view);
      PetscViewerPopFormat(view);
      PetscViewerDestroy(&view);
      exit(0);
    }


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
  std::vector<Mat> sigmas;
  sigmas.resize(L);
  //for (int k=0;k<L;++k) { MatCreateVecs(H, NULL, &sigmas[k]);}
  {
  std::vector<PetscInt> d_nnz(Iend - Istart,1);
  std::vector<PetscInt> o_nnz(Iend - Istart,0);
  for (int k=0;k<L;++k) { MatCreateAIJ(PETSC_COMM_WORLD, Iend - Istart, PETSC_DECIDE,nconf,nconf, 0, d_nnz.data(), 0,
               o_nnz.data(), &(sigmas[k])); } // check &(sigmas[k]
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
                if (config[k]) { MatSetValue(sigmas[k], row_ctr, row_ctr, (PetscScalar)1., ADD_VALUES);}
                else { MatSetValue(sigmas[k], row_ctr, row_ctr, (PetscScalar)-1., ADD_VALUES);}
          }
        }
        row_ctr++;
      }
    }
  }

  for (int k = 0; k < L; ++k) {
  MatSetOption(sigmas[k], MAT_SYMMETRIC, PETSC_TRUE);
  MatSetOption(sigmas[k], MAT_SYMMETRY_ETERNAL, PETSC_TRUE);
  ierr = MatAssemblyBegin(sigmas[k], MAT_FINAL_ASSEMBLY);
  ierr = MatAssemblyEnd(sigmas[k], MAT_FINAL_ASSEMBLY);
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





    if (0 == myrank)
      std::cout << "Processing target " << target
                << " [ epsilon=" << renorm_target << " ] ... ";
    //   EPSReset(eps);

    char* useless_string = new char[100];
    PetscBool eps_interval_set;
    PetscOptionsGetString(NULL, NULL, "-eps_interval", useless_string, 100,&eps_interval_set); 
    if (!(eps_interval_set)) { 
      EPSSetWhichEigenpairs(eps2, EPS_TARGET_REAL);
      EPSSetTarget(eps2, target);
    }
    EPSSetFromOptions(eps2);

    
    //                EPSSetUp(eps);

    ierr = EPSSolve(eps2);  // CHKERRQ(ierr);

    PetscInt nconv = 0;
    ierr = EPSGetConverged(eps2, &nconv);
    CHKERRQ(ierr);
    if (0 == myrank) std::cout << "Solved done. \n";


    Vec use1,use2;
    MatCreateVecs(sigmas[0], NULL, &use1);
    MatCreateVecs(sigmas[0], NULL, &use2);

    if (nconv > 0) {
      ofstream entout;
      ofstream locout;
      ofstream partout;
      ofstream corrout;
      ofstream tcorrout;
      ofstream KLout;
      if (myparameters.measure_KL) {
        myparameters.init_filenames_eigenstate(entout, locout, partout,corrout, tcorrout,KLout,
                                             renorm_target);
      }
      else {
      myparameters.init_filenames_eigenstate(entout, locout, partout,corrout, tcorrout,
                                             renorm_target);
      }

      std::vector<double> energies;
      std::vector<double> rgap;
      PetscScalar Er, Ei;
      Vec Vec_local;
      std::vector<double> Siz(L, 0.);
      std::vector< std::vector<double> > Gij(L);
      for (int k=0;k<L;++k) { Gij[k].resize(L,0.);}
      std::vector< std::vector<double> > Tij(L);
      for (int k=0;k<L;++k) { Tij[k].resize(L,0.);}

      //std::vector<PetscScalar *> all_states;
      //if (myparameters.measure_KL) { if (myrank==0) { all_states.resize(nconv); }}

      std::vector<int> eigenstates_to_follow;
      std::vector<double> energies_to_follow;
      std::vector< std::vector<int> > sites_to_follow;
      std::vector< std::vector<pair<int,int> > > pairs_to_follow;
      std::vector< std::vector<double> > sz_to_follow;

      PetscBool compute_weight=PETSC_TRUE;
      PetscOptionsGetBool(NULL, NULL, "-compute_weight", &compute_weight,NULL); 

      PetscBool measure_everything=PETSC_TRUE;
      PetscOptionsGetBool(NULL, NULL, "-measure_everything", &measure_everything,NULL); 
        PetscBool other_measurements=PETSC_FALSE;
      PetscOptionsGetBool(NULL, NULL, "-other_measurements", &other_measurements,NULL); 


      PetscBool sz_cutoff_set;
        PetscReal sz_cutoff=0.05;
        PetscOptionsGetReal(NULL, NULL, "-sz_cutoff", &sz_cutoff,&sz_cutoff_set); 
        PetscBool sz_product_cutoff_set;
        PetscReal sz_product_cutoff=sz_cutoff*sz_cutoff;
        PetscOptionsGetReal(NULL, NULL, "-sz_product_cutoff", &sz_product_cutoff,&sz_product_cutoff_set); 
        
        sz_cutoff*=2;
        sz_product_cutoff*=4;

      for (int i = 0; i < nconv; i++) {
        ierr = EPSGetEigenpair(eps2, i, &Er, &Ei, xr, NULL);
        CHKERRQ(ierr);
        energies.push_back(PetscRealPart(Er));

        if (sz_product_cutoff_set) { sz_cutoff_set=PETSC_TRUE;}

        std::vector< pair<int,int> > prediction_strong_correl_pair; prediction_strong_correl_pair.resize(0);
        std::vector< int > prediction_site; prediction_site.resize(0);
        std::vector<double> sz(L,0.);
        if (sz_cutoff_set) {
        for (int k=0;k<L;++k) {
          MatMult(sigmas[k],xr,use1);
          VecDot(use1,xr,&sz[k]);
          //std::cout << "Sz " << k << " " << sz[k] << " " << Er << endl;
        }
        
        for (int j=0;j<L;++j)
          { if (fabs(sz[j])<sz_cutoff)
                { prediction_site.push_back(j);
                  //std::cout << j << " passes the deal " << Er << endl;
                for (int k=j+1;k<L;++k)
                    { if ( (fabs(sz[k])<sz_cutoff) && ( (fabs(sz[k]*sz[j])<sz_product_cutoff) ) )
                      { //std::cout << "together with " << k << " " << Er << endl;
                        
                        prediction_strong_correl_pair.push_back(make_pair(j,k));}
                    }
                }
          }
          bool prediction_strong_correl_found=PETSC_FALSE;
          if (prediction_strong_correl_pair.size()!=0) {
            prediction_strong_correl_found=PETSC_TRUE;
            std::cout << "### Prediction strong correl for Eigenstate with energy " << Er << endl;
            for (int ss=0;ss<prediction_strong_correl_pair.size();++ss) {
            std::cout << "### Prediction strong correl for pair : " << prediction_strong_correl_pair[ss].first << " " << prediction_strong_correl_pair[ss].second << endl;}
          }

        }

        if (prediction_site.size()!=0) { eigenstates_to_follow.push_back(i); sites_to_follow.push_back(prediction_site); 
        energies_to_follow.push_back(Er); sz_to_follow.push_back(sz);
        if (prediction_strong_correl_pair.size()!=0) { pairs_to_follow.push_back(prediction_strong_correl_pair);}
        }
      

            if (compute_weight)
          {  PetscInt number_of_weight_cutoff_values=9;
        PetscOptionsGetInt(NULL, NULL, "-cutoff_values", &number_of_weight_cutoff_values,NULL); 
            std::vector<double>  weight_cutoff(number_of_weight_cutoff_values,0.);
           for (int c=0;c<number_of_weight_cutoff_values;++c) { weight_cutoff[c]=(c+1)*0.25/(number_of_weight_cutoff_values+1);}
          std::vector< std::vector<double> > weight_at_cutoff_at_range;
          weight_at_cutoff_at_range.resize(number_of_weight_cutoff_values);

          for (int c=0;c<number_of_weight_cutoff_values;++c) { weight_at_cutoff_at_range[c].resize(L/2+1);}
          std::vector<double> Normalization(L/2+1,0.);
          for (int k=0;k<L;++k) {
              MatMult(sigmas[k],xr,use1);
              for (int range=1;range<=(L/2);++range) {
                  MatMult(sigmas[(k+range)%L],use1,use2); // pbc assumed here - TODO : change for obc
                  double C;
                  VecDot(use2,xr,&C);
                  C=0.25*fabs(C-sz[k]*sz[(k+range)%L]);
                  Normalization[range]+=1.0;
                //  std::cout << "W: " << k << " " << range << " C=" << C << endl;
                  for (int c=0;c<number_of_weight_cutoff_values;c++)
                  {// cout << " Testing (r=" << range << " c=" << c << "with weight : " << weight_cutoff[c];
                   
                    if (C>weight_cutoff[c]) {/*cout << "Passing ... \n"; */weight_at_cutoff_at_range[c][range]+=1.0;}
                      else {/*cout <<"\n"; */break;}
                  }
              }
          }

        cout << "Weight "; for (int c=0;c<number_of_weight_cutoff_values;++c) { cout << weight_cutoff[c] << " ";} cout << endl;
        for (int range=1;range<=(L/2);++range) {
          cout << "Weight-range " << range << " ";
           for (int c=0;c<number_of_weight_cutoff_values;++c) { cout << weight_at_cutoff_at_range[c][range]/Normalization[range] << " ";} cout << endl;
        }
      }
      }
      int ll=0;
      for (std::vector<int>::iterator it=eigenstates_to_follow.begin();it!=eigenstates_to_follow.end();++it) {
        EPSGetEigenpair(eps2, *it, &Er, &Ei, xr, NULL);

        if (measure_everything) {
        std::vector<double> sz=sz_to_follow[ll];
        for (int k=0;k<L;++k) {
          cout << "Sz " << k << " " << sz[k] << " " << energies_to_follow[ll] << endl;
          MatMult(sigmas[k],xr,use1);
          
          std::vector<double> szkp(L-k-1);
            for (int pp=k+1;pp<L;++pp)
              { 
                MatMult(sigmas[pp],use1,use2);
                VecDot(use2,xr,&szkp[pp-k-1]);
                cout << "SzSz " << k << " " << pp << " " << 0.25*(szkp[pp-k-1]-sz[k]*sz[pp]) << " " << Er << endl;
              }
        }
                 
        }
        else {  // measure only guessed correlations
        int s=sites_to_follow[ll].size();
        std::vector<double> sz=sz_to_follow[ll];
        for (int si=0;si<s;++si) {
          int k=sites_to_follow[ll][si];
          cout << "Sz " << k << " " << sz[k] << " " << energies_to_follow[ll] << endl;
          MatMult(sigmas[k],xr,use1);
          std::vector<double> szkp(s-si-1);
          for (int pp=si+1;pp<s;++pp)
              { MatMult(sigmas[sites_to_follow[ll][pp]],use1,use2);
                VecDot(use2,xr,&szkp[pp-k-1]);
                cout << "SzSz " << k << " " << sites_to_follow[ll][pp] << " " << 0.25*(szkp[pp-k-1]-sz[k]*sz[sites_to_follow[ll][pp]]) << " " << Er << endl;
              }
        }
        }

        double pi; 
          double local_S1=0.;
          for (int row_ctr = Istart; row_ctr<Iend;++row_ctr) {
            VecGetValues( xr, 1, &row_ctr, &pi );
             if ((pi != 0)) {
             local_S1 -= 2.0* pi * pi * log(pi);
              }
          }
          double global_S1=0.;
          MPI_Reduce(&local_S1, &global_S1, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);
          cout << "S1 " << global_S1 << " " << energies_to_follow[ll] << endl;

        // measure KL, and < n | sigma_i | m > with other states
        int llj=0; double Er2;
        for (std::vector<int>::iterator jt=eigenstates_to_follow.begin();jt!=eigenstates_to_follow.end();++jt) {
          if (it!=jt) {
          EPSGetEigenpair(eps2, *jt, &Er2, &Ei, use1, NULL);
          double pi; double qi;
          double local_KL=0.;
          for (int row_ctr = Istart; row_ctr<Iend;++row_ctr) {
            VecGetValues( xr, 1, &row_ctr, &pi );
            VecGetValues( use1, 1, &row_ctr, &qi );
             if ((pi != 0) && (qi != 0)) {
             local_KL += 2.0 * pi * pi * log(pi / qi);
              }
          }
          double global_KL;
          MPI_Reduce(&local_KL, &global_KL, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);
          //MPI_AllReduce(&local_KL, &global_KL, 1, MPI_double, MPI_SUM, PETSC_COMM_WORLD);
          // myrank=0
          if (myrank==0) {
          cout << "KL " << global_KL << " " << energies_to_follow[ll] << " " <<  Er2 << endl;
          }

          if (measure_everything) {
              std::vector<double> sigma_indicator(L,0);
            for (int k=0;k<L;++k) {
              MatMult(sigmas[k],xr,use2);
              VecDot(use2,use1,&sigma_indicator[k]);
            }
            for (int k=0;k<L;++k) {
              std::cout << "Sigmaindic " << k << " " << sigma_indicator[k] << " " << energies_to_follow[ll] << " " <<  Er2 << endl;
            }
          }
          else {
            int s=sites_to_follow[ll].size();
            std::vector<double> sigma_indicator(s,0.);
            for (int si=0;si<s;++si) {
            int k=sites_to_follow[ll][si];
              MatMult(sigmas[k],xr,use2);
              VecDot(use2,use1,&sigma_indicator[si]);
            }
            for (int si=0;si<s;++si) {
              std::cout << "Sigmaindic " << sites_to_follow[ll][si] << " " << sigma_indicator[si] << " " << energies_to_follow[ll] << " " <<  Er2 << endl;
            }
          }
          }
      llj++;
        }

        ll++;
        // other measurements on eigenstate to follow ...
        PetscBool other_measurements=PETSC_FALSE;
        if (other_measurements) {
           /*
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
                tcorrout << r << " " << " " << s << " " << Tij[r][s] << endl;
            } }
          }
          if (myparameters.measure_entanglement) {
            myobservable.compute_entanglement_spectrum(state);
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
        */
      }
      }

    } // nconv>0
/*
      entout.close();
      locout.close();
      partout.close();
      corrout.close();
      tcorrout.close();
      KLout.close();
*/
      if (myrank == 0) {
        ofstream enout;
        ofstream rgapout;
        myparameters.init_filenames_energy(enout, rgapout, renorm_target);
        // as a header add info about the min/max energies
        enout << "# (Emin, Emax) = " << Eminc << "," << Emaxc << endl;
        enout << "# Etarget = " << target << endl;
        enout << "# Erenorm_target = " << renorm_target << endl;
        
        for (int i = 0; i < nconv; i++) {
          cout << energies[i] << "\n";
          enout << energies[i] << endl;
        }
        
        std::sort(energies.begin(), energies.end());

        for (int r = 1; r < (nconv - 1); ++r) {
          double e1 = energies[r];
          double g0 = e1 - energies[r - 1];
          double g1 = energies[r + 1] - e1;
          if (g0 > g1) {
            rgap.push_back(g1 / g0);
          } else {
            rgap.push_back(g0 / g1);
          }
        }
        for (int i = 0; i < rgap.size(); ++i) {
          rgapout << rgap[i] << endl;
        }

        rgapout.close();
        enout.close();
      }
    
  }// target
  SlepcFinalize();
  return 0;
}