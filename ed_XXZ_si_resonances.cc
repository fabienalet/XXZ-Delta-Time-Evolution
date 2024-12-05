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





   // if (0 == myrank)
   //   std::cout << "Processing target " << target
   //             << " [ epsilon=" << renorm_target << " ] ... ";
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

    double Cmax=0.; double E_Cmax=0.; int site1_Cmax=-1; int site2_Cmax=-1;

    if (nconv > 0) {
      ofstream entout;
      ofstream locout;
      ofstream partout;
      ofstream corrout;
      ofstream tcorrout;
      ofstream KLout;
      ofstream sigmaout;
      if (myparameters.measure_KL) { myparameters.init_filename_KL(KLout);}
      if (myparameters.measure_local) { myparameters.init_filename_local(locout);}
      if (myparameters.measure_correlations) { myparameters.init_filename_correlations(corrout);}
      if (myparameters.measure_participation) { myparameters.init_filename_participation(partout);}
      if (myparameters.measure_entanglement) { myparameters.init_filename_entanglement(entout);}

  
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

      PetscBool debug=PETSC_FALSE;
      PetscOptionsGetBool(NULL, NULL, "-debug", &debug,NULL);

      PetscBool compute_weight=PETSC_TRUE;
      
      PetscOptionsGetBool(NULL, NULL, "-compute_weight", &compute_weight,NULL); 

      PetscBool measure_Cmax=PETSC_TRUE;
      PetscOptionsGetBool(NULL, NULL, "-measure_Cmax", &measure_Cmax,NULL);
      if (measure_Cmax) { compute_weight=PETSC_TRUE;}
      

      PetscInt number_of_weight_cutoff_values=9; 
      PetscOptionsGetInt(NULL, NULL, "-cutoff_values", &number_of_weight_cutoff_values,NULL);
      std::vector< std::vector<double> > weight_at_cutoff_at_range;
      std::vector<double> Normalization;
      std::vector<double> weight_cutoff;


        if (compute_weight) {
        weight_cutoff.resize(number_of_weight_cutoff_values,0.);
        for (int c=0;c<number_of_weight_cutoff_values;++c) { weight_cutoff[c]=(c+1)*0.25/(number_of_weight_cutoff_values+1);}
        weight_at_cutoff_at_range.resize(number_of_weight_cutoff_values);
        for (int c=0;c<number_of_weight_cutoff_values;++c) { weight_at_cutoff_at_range[c].resize(L/2+1);}
        Normalization.resize(L/2+1,0.);
        }
 
      PetscInt min_range=0;
      PetscOptionsGetInt(NULL, NULL, "-min_range", &min_range,NULL); 


      PetscBool measure_everything=PETSC_FALSE;
      PetscOptionsGetBool(NULL, NULL, "-measure_everything", &measure_everything,NULL); 

      PetscBool measure_sigma_indicator=PETSC_TRUE;
      PetscOptionsGetBool(NULL, NULL, "-measure_sigma", &measure_sigma_indicator,NULL); 
      if (measure_sigma_indicator) { myparameters.init_filename_sigma(sigmaout);}


    
      PetscBool other_measurements=PETSC_FALSE;
      PetscOptionsGetBool(NULL, NULL, "-other_measurements", &other_measurements,NULL); 


        PetscBool sz_cutoff_set=PETSC_TRUE;
        PetscReal sz_cutoff=0.05;
        PetscOptionsGetReal(NULL, NULL, "-sz_cutoff", &sz_cutoff,&sz_cutoff_set); 
        PetscBool C_cutoff_set;
        PetscReal C_cutoff=0.25-sz_cutoff*sz_cutoff;
        PetscOptionsGetReal(NULL, NULL, "-C_cutoff", &C_cutoff,&C_cutoff_set); 
       
        if (!(sz_cutoff_set)) { sz_cutoff=sqrt(0.25-C_cutoff);}
        if (C_cutoff_set) { sz_cutoff_set=PETSC_TRUE;}
        //if (sz_cutoff_set) { C_cutoff_set=PETSC_TRUE; }

        sz_cutoff*=2;
        C_cutoff*=4;
        if (!(C_cutoff_set) && (!(sz_cutoff_set)) ) {
          if (myrank==0) { cout << "No cutoff set, exiting\n";}
          exit(0);
        }
      for (int i = 0; i < nconv; i++) {
        ierr = EPSGetEigenpair(eps2, i, &Er, &Ei, xr, NULL);
        CHKERRQ(ierr);
        energies.push_back(PetscRealPart(Er));

        std::vector< pair<int,int> > prediction_strong_correl_pair; prediction_strong_correl_pair.resize(0);
        std::vector< int > prediction_site; prediction_site.resize(0);
        std::vector<double> sz(L,0.);
     //   if (sz_cutoff_set) || (sz_product) {
        for (int k=0;k<L;++k) {
          MatMult(sigmas[k],xr,use1);
          VecDot(use1,xr,&sz[k]);
         // if (myparameters.measure_local) { locout << k << " " << sz[k] << " " << Er << endl; } // TODO maybe only for special sites ? second loop instead of here ?
        }
        
        // TODO : get pbc back !
        PetscBool pbc=PETSC_TRUE;
        PetscOptionsGetBool(NULL, NULL, "-pbc", &pbc, NULL);

        for (int j=0;j<L;++j)
          { if (fabs(sz[j])<sz_cutoff)
                { //prediction_site.push_back(j);
                  //std::cout << j << " passes the deal " << Er << endl;
                for (int k=j+1;k<L;++k)
                    { if ( (fabs(sz[k])<sz_cutoff) && ( (fabs(sz[k]*sz[j])<(0.25-C_cutoff)) ) )
                      { //std::cout << "together with " << k << " " << Er << endl;
                        if ( ((k-j)>min_range) && ((!(pbc)) || ((j+L-k)>min_range)) )
                        {  prediction_strong_correl_pair.push_back(make_pair(j,k));
                          prediction_site.push_back(j); prediction_site.push_back(k);
                        }
                      }
                    }
                }
          }
          bool prediction_strong_correl_found=PETSC_FALSE;
          if (prediction_strong_correl_pair.size()!=0) {
            prediction_strong_correl_found=PETSC_TRUE;
         if(debug)   if (myrank==0) std::cout << "### Prediction strong correl for Eigenstate with energy " << Er << endl;
            for (int ss=0;ss<prediction_strong_correl_pair.size();++ss) {
          if(debug)  if (myrank==0) std::cout << "### Prediction strong correl for pair : " << prediction_strong_correl_pair[ss].first << " " << prediction_strong_correl_pair[ss].second << endl; //}
          }
          }
        
        // remove duplicates of sites
        sort(prediction_site.begin(), prediction_site.end());
        auto it = unique(prediction_site.begin(), prediction_site.end());
        prediction_site.erase(it, prediction_site.end());

        if (prediction_strong_correl_pair.size()!=0) {
          eigenstates_to_follow.push_back(i);
          energies_to_follow.push_back(Er); 
          pairs_to_follow.push_back(prediction_strong_correl_pair);
          sz_to_follow.push_back(sz);
          sites_to_follow.push_back(prediction_site);
        }
      


            if (compute_weight)
          {  double C; 
          for (int k=0;k<L;++k) {
              MatMult(sigmas[k],xr,use1);
              for (int range=1;range<=(L/2);++range) {
                  MatMult(sigmas[(k+range)%L],use1,use2); // pbc assumed here - TODO : change for obc           
                  VecDot(use2,xr,&C);
                  C=0.25*fabs(C-sz[k]*sz[(k+range)%L]);
                  if (measure_Cmax) { if (C>Cmax) { E_Cmax=Er; Cmax=C; site1_Cmax=k; site2_Cmax=(k+range)%L;} }
                  Normalization[range]+=1.0;
                //  std::cout << "W: " << k << " " << range << " C=" << C << endl;
                  for (int c=0;c<number_of_weight_cutoff_values;c++)
                  {
                    
                    if (C>weight_cutoff[c]) {/*cout << "Passing ... \n"; */weight_at_cutoff_at_range[c][range]+=1.0;}
                      else {/*cout <<"\n"; */break;}
                  }
              }
          }
      }
      }
      
      int ll=0;
      if (myrank==0) { cout << "*** Measuring on " << eigenstates_to_follow.size() << " eigenstates to follow\n";}



      


      for (std::vector<int>::iterator it=eigenstates_to_follow.begin();it!=eigenstates_to_follow.end();++it) {
        if (!(ll%100)) { if (myrank==0) { cout << ll << " eigenstates done\n";}}
        EPSGetEigenpair(eps2, *it, &Er, &Ei, xr, NULL);

        int ss=sites_to_follow[ll].size();
        std::vector< Vec > applications_of_sites_to_follow(ss);
        if (!(measure_everything)) {
          for (int si=0;si<ss;++si) { 
          MatCreateVecs(sigmas[0], NULL, &applications_of_sites_to_follow[si]);
          int k=(int) sites_to_follow[ll][si];
          MatMult(sigmas[(int) sites_to_follow[ll][si]],xr,applications_of_sites_to_follow[si]);
        }
        }

        if (measure_everything) {
        std::vector<double> sz=sz_to_follow[ll];
        if (myparameters.measure_local) { 
        for (int k=0;k<L;++k) { locout << k << " " << sz[k] << " " << Er << endl; } 
        }
        for (int k=0;k<L;++k) {
//          cout << "Sz " << k << " " << sz[k] << " " <<  << endl;
          MatMult(sigmas[k],xr,use1);
          
          std::vector<double> szkp(L-k-1);
            for (int pp=k+1;pp<L;++pp)
              { 
                MatMult(sigmas[pp],use1,use2);
                VecDot(use2,xr,&szkp[pp-k-1]);
                if (myparameters.measure_correlations) {
	              corrout << k+1 << " " << pp+1 << " " << 0.25*(szkp[pp-k-1]-sz[k]*sz[pp]) << " " << Er << endl;
              }
              }
        }
                 
        }
        else {  // measure only guessed correlations
        /*
        int s=sites_to_follow[ll].size();
        std::vector<double> sz=sz_to_follow[ll];
        for (int si=0;si<s;++si) {
          int k=(int) sites_to_follow[ll][si];
          MatMult(sigmas[k],xr,use1);
          if (s>1)  {
          std::vector<double> szkp(s-si-1);
          for (int pp=si+1;pp<s;++pp)
              { 
                int p=(int) sites_to_follow[ll][pp];
                MatMult(sigmas[p],use1,use2);
                VecDot(use2,xr,&szkp[pp-si-1]);
                if (myparameters.measure_correlations) {
	              corrout << k+1 << " " << p+1 << " " << 0.25*(szkp[pp-si-1]-sz[k]*sz[p]) << " " << Er << endl;
              }
              }
          }
        }
        }
        */
        std::vector<double> sz=sz_to_follow[ll];
        for (int si=0;si<ss;++si) {
          int k=(int) sites_to_follow[ll][si];
          if (ss>1)  {
          std::vector<double> szkp(ss-si-1);
          for (int pp=si+1;pp<ss;++pp)
              { int p=(int) sites_to_follow[ll][pp];
                MatMult(sigmas[p],applications_of_sites_to_follow[si],use2);
                VecDot(use2,xr,&szkp[pp-si-1]);
                if (myparameters.measure_correlations) {
	              corrout << k+1 << " " << p+1 << " " << 0.25*(szkp[pp-si-1]-sz[k]*sz[p]) << " " << Er << endl;
              }
              }
          }
        }
        }



        double pi; 
        if (myparameters.measure_participation) {
          double local_S1=0.;
          for (int row_ctr = Istart; row_ctr<Iend;++row_ctr) {
            VecGetValues( xr, 1, &row_ctr, &pi );
             if ((pi != 0)) { local_S1 -= 2.0* pi * pi * log(fabs(pi));}
          }
          double global_S1=0.;
          MPI_Reduce(&local_S1, &global_S1, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);
          partout << "S1 " << global_S1 << " " << energies_to_follow[ll] << endl;
        }
        // measure KL, and < n | sigma_i | m > with other states
        int llj=0; double Er2;
        if ((myparameters.measure_KL) || (measure_sigma_indicator)) {
        //for (std::vector<int>::iterator jt=eigenstates_to_follow.begin();jt!=eigenstates_to_follow.end();++jt) {
        std::vector<int>::iterator ending=eigenstates_to_follow.end();
        //if (measure_nearby_eigenstates) { ending=it+}
          for (std::vector<int>::iterator jt=(it+1);jt!=eigenstates_to_follow.end();++jt) {
        //  if (it!=jt) {
          EPSGetEigenpair(eps2, *jt, &Er2, &Ei, use1, NULL);
          if (myparameters.measure_KL) {
          //  if (myparameters.measure_all_KL) {

          double pi; double qi;
          double local_KL=0.;
          double local_KL2=0.;
          for (int row_ctr = Istart; row_ctr<Iend;++row_ctr) {
            VecGetValues( xr, 1, &row_ctr, &pi );
            VecGetValues( use1, 1, &row_ctr, &qi );
             if ((pi != 0) && (qi != 0)) {
             local_KL += 2.0 * pi * pi * log(fabs(pi / qi));
             local_KL2 += 2.0 * qi * qi * log(fabs(qi / pi));
              }
          }
          double global_KL;
          double global_KL2;
          MPI_Reduce(&local_KL, &global_KL, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);
          MPI_Reduce(&local_KL2, &global_KL2, 1, MPI_DOUBLE, MPI_SUM, 0,PETSC_COMM_WORLD);
          //MPI_AllReduce(&local_KL, &global_KL, 1, MPI_double, MPI_SUM, PETSC_COMM_WORLD);
          // myrank=0
          if (myrank==0) {
          //KLout << global_KL << " " << energies_to_follow[ll] << " " <<  Er2 << endl;
          KLout << global_KL << " " << global_KL2 << " " << energies_to_follow[ll] << " " <<  Er2 << endl;
          }
            }
            // }
          if (measure_sigma_indicator) {
            if (measure_everything) {
              std::vector<double> sigma_indicator(L,0);
            for (int k=0;k<L;++k) {
              MatMult(sigmas[k],xr,use2);
              VecDot(use2,use1,&sigma_indicator[k]);
            }
            for (int k=0;k<L;++k) {
              sigmaout << "Sig " <<  k << " " << sigma_indicator[k] << " " << energies_to_follow[ll] << " " <<  Er2 << endl;
            }
          } 
          else {
            std::vector<double> sigma_indicator(ss,0.);
            for (int si=0;si<ss;++si) {
              VecDot(applications_of_sites_to_follow[si],use1,&sigma_indicator[si]);
              sigmaout << "Sig " <<  (int) sites_to_follow[ll][si] << " " << sigma_indicator[si] << " " << energies_to_follow[ll] << " " <<  Er2 << endl;
            }
          }
          }
       //   } it!=jt
      llj++;
        }
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
        */
      }
      
      } // loop over states

    if (myparameters.measure_KL) {KLout.close();}
      if (myparameters.measure_local) { locout.close();}
      if (myparameters.measure_correlations) { corrout.close();}
      if (myparameters.measure_participation) { partout.close();}
      if (myparameters.measure_entanglement) { entout.close();}
       if (measure_sigma_indicator) {sigmaout.close();}



      if (myrank == 0) {
          if (compute_weight) { 
           ofstream weightout;
           myparameters.init_filename_weight(weightout);
           weightout << "#Weight "; for (int c=0;c<number_of_weight_cutoff_values;++c) { weightout << weight_cutoff[c] << " ";} weightout << endl;
          for (int range=1;range<=(L/2);++range) {
            weightout << "#Weight-range " << range << " ";
            for (int c=0;c<number_of_weight_cutoff_values;++c) { weightout << weight_at_cutoff_at_range[c][range]/Normalization[range] << " ";} weightout << endl;
          }
            weightout.close();

            if (measure_Cmax) {
              ofstream Cmaxout;
              myparameters.init_filename_Cmax(Cmaxout);
              Cmaxout << Cmax << " " << site1_Cmax << " " << site2_Cmax << " " << E_Cmax << endl;
              Cmaxout.close();

            }
            }


 	PetscBool interval_set=PETSC_FALSE;
 	char* eps_interval_string = new char[1000];
   	PetscOptionsGetString(NULL, NULL, "-eps_interval", eps_interval_string, 1000,
                                &interval_set);  // CHKERRQ(ierr);

        ofstream enout;
        ofstream rgapout;
        myparameters.init_filenames_energy(enout, rgapout, renorm_target);
        // as a header add info about the min/max energies
        enout << "# (Emin, Emax) = " << Eminc << "," << Emaxc << endl;
        if (interval_set) { enout << "# E_interval " << eps_interval_string << endl;}
        else { enout << "# Etarget = " << target << endl; }
        enout << "# nconv = " << nconv << endl;
        
        for (int i = 0; i < nconv; i++) {
    //      cout << energies[i] << "\n";
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
        if (energies_to_follow.size()) {
        enout << "### Special energies " << ((double) energies_to_follow.size()/nconv)*100 << " % \n";
        int ll=0;
        for (std::vector<double>::iterator it=energies_to_follow.begin();it!=energies_to_follow.end();++it) {
            enout << *it << endl;
        }
        }
        rgapout.close();
        enout.close();
      }
      
    } // nconv>0
  }// target
  SlepcFinalize();
  return 0;
}