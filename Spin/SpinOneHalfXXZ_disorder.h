#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

class Hamiltonian {
  // Simulating H = \sum_i (J_i S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + Delta S_i^z S_{i+1}^z - h_i S_i^z)
 private:
  basis *basis_pointer;
  Mat *pointer_to_H;

 public:
  Hamiltonian() {
    // by default we just store the input parameters (this is used by
    // ProductState to retrieve them)
    std::cout << "Entering default constructor" << std::endl;
    get_parameters();
    std::cout << "Finished reading entry parameters" << std::endl;
  };
  Hamiltonian(basis *this_basis_pointer, Mat *matrix_pointer);
  ~Hamiltonian() {}

  void get_parameters();

  void init_matrix_sizes(PetscInt Istart, PetscInt Iend);
  void create_matrix(PetscInt Istart, PetscInt Iend);
  std::vector<unsigned long int> get_neighbors(unsigned long int);
  void create_matrix_lapack(double *A);
  void diagonalize_lapack(double *A, double w[], bool eigenvectors);
  void print_local_fields();
  std::string string_names;

  int L;
  int LA;
  int LB;
  double Delta;
  double field_default;
  double disorder;
  double J;
  std::vector<double> field;
  std::vector<double> coupling;
  PetscBool pbc;
  PetscBool gaussian_field;
  PetscBool speckle_field;
  PetscBool red_speckle;
  PetscBool box_field;
  PetscBool random_bond_powerlaw;

  double bigshift;
  PetscBool comparison_matrix;


  PetscInt seed;

  int myrank;
  short int NUMBER_OF_STATES;
};

Hamiltonian::Hamiltonian(basis *this_basis_pointer, Mat *matrix_pointer) {
  basis_pointer = this_basis_pointer;
  pointer_to_H = matrix_pointer;
  L = basis_pointer->L;
  LA = basis_pointer->LA;
  LB = basis_pointer->LB;
  myrank = basis_pointer->myrank;
  NUMBER_OF_STATES = basis_pointer->NUMBER_OF_STATES;
}

void Hamiltonian::get_parameters() {
  J = 1.0;
  disorder = 0.;
  Delta = 1.;
  field_default = 0.;
  pbc = PETSC_FALSE;
  box_field = PETSC_TRUE;
  gaussian_field = PETSC_FALSE;
  speckle_field = PETSC_FALSE;
  red_speckle = PETSC_FALSE;
  random_bond_powerlaw = PETSC_FALSE;
  seed = 20213006;
  bigshift = 0;
  comparison_matrix = PETSC_FALSE;

  PetscErrorCode ierr;
  L = 6;
  ierr = PetscOptionsGetInt(NULL, NULL, "-L", &L, NULL);  // CHKERRQ(ierr);
  int Ldouble = 2 * L;
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ldouble", &Ldouble, NULL);
  L = Ldouble / 2;

  ierr = PetscOptionsGetReal(NULL, NULL, "-J", &J, NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-Delta", &Delta,
                             NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-random_bond", &random_bond_powerlaw,
                             NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-field_default", &field_default,
                             NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-speckle_field", &speckle_field,
                             NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-red", &red_speckle,
                             NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-gaussian_field", &gaussian_field,
                             NULL);                            // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-pbc", &pbc, NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-comparison", &comparison_matrix, NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-disorder", &disorder,
                             NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-K", &bigshift,
                             NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-bigshift", &bigshift, NULL);
  ierr =
      PetscOptionsGetInt(NULL, NULL, "-seed", &seed, NULL);  // CHKERRQ(ierr);
  coupling.resize(L, J);
  field.resize(L, field_default);
  if (!(pbc)) {
    coupling[L - 1] = 0.;
  }
  if (speckle_field && gaussian_field) {
    std::cerr << "Can't have two types of random disorder (speckle and "
                 "gaussian) !!\n";
    exit(0);
  }
  if (speckle_field || gaussian_field) {
    box_field = PETSC_FALSE;
  }
  if (random_bond_powerlaw) {
    box_field = PETSC_FALSE;
  }

  if (disorder > 0) {
    if (random_bond_powerlaw) {
      if (myrank == 0) {
        std::cout << "# Choosing power-law distribution for bonds, power-law "
                     "exponent = "
                  << disorder << "\n";
      }

      // power law distribution (between 0 and 1) P(J)=1/h * J^(-1+1/h)
      // h between 0 and infty
      boost::mt19937 generator;
      generator.seed(seed);
      boost::random::uniform_real_distribution<double> uniform_box(0., 1.);
      if (myrank == 0) {
        std::cout << "# bonds= {";
      }
      int Lmax = L;
      if (!(pbc)) {
        Lmax = L - 1;
      }
      for (size_t i = 0; i < Lmax; i++) {  // power-law rule
        coupling[i] = pow(uniform_box(generator), disorder);
      }
      for (size_t i = 0; i < L; i++) {
        if (myrank == 0) {
          std::cout << coupling[i] << " ";
        }
      }
    }

    if (box_field) {
      PetscBool without_boost_rng=PETSC_FALSE;
      ierr = PetscOptionsGetBool(NULL, NULL, "-without_boost_rng", &without_boost_rng,
                                 NULL);


      if (without_boost_rng) {
        std::random_device device;
        std::mt19937 generator(device());
        generator.seed(seed);
        std::uniform_real_distribution<double> box(-disorder, disorder);
        for (int i = 0; i < L; i++) {
          field[i] += box(generator);
        }
      }
      else {
      boost::mt19937 generator;
      generator.seed(seed);
      boost::random::uniform_real_distribution<double> box(-disorder, disorder);
      for (int i = 0; i < L; i++) {
        field[i] += box(generator);
      }
    }
    }
    if (gaussian_field) {
      boost::mt19937 generator;
      generator.seed(seed);
      boost::random::normal_distribution<double> gaussian(field_default,
                                                          disorder);
      if (myrank == 0) {
        std::cout << "# Choosing gaussian disorder with mean value "
                  << field_default << " and standard deviation " << disorder
                  << std::endl;
      }
      for (int i = 0; i < L; i++) {
        field[i] += gaussian(generator);
      }
    }
    if (speckle_field) {
      boost::mt19937 generator;
      generator.seed(seed);
      boost::random::exponential_distribution<double> speckle(1. / disorder);
      if (myrank == 0) {
        std::cout << "# Choosing speckle disorder ";
        if (red_speckle)
          std::cout << "(red)";
        else
          std::cout << "(blue)";
        std::cout << " with mean value " << field_default
                  << " and V0= " << disorder << std::endl;
      }
      for (int i = 0; i < L; i++) {
        double a = -disorder + speckle(generator);
        if (red_speckle) {
          a = -a;
        }
        field[i] = a;
      }
    }
  }


  PetscBool fields_string_set=PETSC_FALSE;
  char* fields_c_string = new char[10000];
  ierr = PetscOptionsGetString(NULL, NULL, "-fields", fields_c_string, 10000,
                               &fields_string_set);  
  if (fields_string_set) {
    field.resize(0);
    std::string fields_string(fields_c_string);
    std::stringstream fieldstr;
    fieldstr.str(fields_string);
    double ss;
    while (fieldstr >> ss) {
      field.push_back(ss);
    }
    if (field.size()!=L) { std::cout << "Error !! Too few fields !!!\n"; exit(0);}
    delete[] fields_c_string;
  } 


  if (myrank == 0) {
    std::cout << "# field= { ";
    for (int i = 0; i < L; i++) {
      std::cout << field[i] << " ";
    }
    std::cout << " }" << endl;

    std::cout << "# seed " << seed << endl;

  }

  std::stringstream ss;
  ss << ".Delta=" << Delta;
  if (J != 1) {
    ss << ".J=" << J;
  }
  if (disorder > 0) {
    ss << ".h=" << disorder;
  } else {
    ss << ".field_default=" << field_default;
  }

  if (pbc)
    ss << ".pbc";
  else
    ss << ".open";

  if (disorder > 0) {
    ss << ".seed=" << seed;
  }
  string_names = ss.str();
}

void Hamiltonian::init_matrix_sizes(PetscInt Istart, PetscInt Iend) {
  std::vector<PetscInt> d_nnz;
  std::vector<PetscInt> o_nnz;
  size_t this_size = Iend - Istart;
  d_nnz.resize(this_size, 0);
  o_nnz.resize(this_size, 0);

  int row_ctr = 0;
  for (int nsa = 0; nsa < basis_pointer->valid_sectors; ++nsa) {
    //   int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      std::vector<unsigned short int> config(L, 0);
      std::vector<unsigned short int> confA =
          basis_pointer->Confs_in_A[nsa][ca];
      for (int r = 0; r < LA; ++r) {
        config[r] = confA[r];
      }
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        //    cout << " *** cb=" << cb << endl;
        if ((row_ctr >= Istart) && (row_ctr < Iend)) {
          // construct B-part only if in the good range ...
          std::vector<unsigned short int> confB =
              basis_pointer->Confs_in_B[nsa][cb];
          for (int r = 0; r < LB; ++r) {
            config[r + LA] = confB[r];
          }

          std::vector<unsigned short int> newconfig = config;
          std::vector<unsigned short int> newconfA = confA;
          std::vector<unsigned short int> newconfB = confB;
          int j;
          double diag = 0.;
          for (int r = 0; r < L; ++r) {
            if (coupling[r]) {
              int rb = r;
              int rb2 = (r + 1) % L;
              if (config[rb] == config[rb2]) {
                // Only diagonal term
                diag += 0.25 * Delta * coupling[r];
              } else {
                diag -= 0.25 * Delta * coupling[r];
                newconfig[rb] = 1 - config[rb];
                newconfig[rb2] = 1 - config[rb2];

                int nleft = 0;
                int nright = 0;
                for (int p = 0; p < LA; ++p) {
                  newconfA[p] = newconfig[p];
                  nleft += newconfig[p];
                }
                for (int p = 0; p < LB; ++p) {
                  newconfB[p] = newconfig[p + LA];
                  nright += newconfig[p + LA];
                }

                int new_nsa = basis_pointer->particle_sector[std::make_pair(
                    nleft,
                    nright)];  // int
                               // new_nsb=basis_pointer->partner_sector[new_nsa];
                newconfig[rb] = config[rb];
                newconfig[rb2] = config[rb2];
                j = basis_pointer->starting_conf[new_nsa] +
                    basis_pointer->InverseMapA[new_nsa][newconfA] *
                        basis_pointer->Confs_in_B[new_nsa].size() +
                    basis_pointer->InverseMapB[new_nsa][newconfB];
                if ((j >= Iend) or (j < Istart)) {
                  o_nnz[row_ctr - Istart]++;
                } else {
                  d_nnz[row_ctr - Istart]++;
                }
              }
            }
            // field part
            if (config[r]) {
              diag += field[r] * 0.5;
            } else {
              diag -= field[r] * 0.5;
            }
          }
          diag += bigshift;
          if (diag != 0) {
            d_nnz[row_ctr - Istart]++;
          }
        }
        row_ctr++;
      }  // loop over cb
    }    // over ca
  }      // over nsA sectors

  MatCreateAIJ(PETSC_COMM_WORLD, Iend - Istart, PETSC_DECIDE,
               basis_pointer->total_number_of_confs,
               basis_pointer->total_number_of_confs, 0, d_nnz.data(), 0,
               o_nnz.data(), &(*pointer_to_H));
}

void Hamiltonian::create_matrix(PetscInt Istart, PetscInt Iend) {
  int row_ctr = 0;
  int j = 0;
  for (int nsa = 0; nsa < basis_pointer->valid_sectors; ++nsa) {
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      std::vector<unsigned short int> config(L, 0);
      std::vector<unsigned short int> confA =
          basis_pointer->Confs_in_A[nsa][ca];
      for (int r = 0; r < LA; ++r) {
        config[r] = confA[r];
      }
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        if ((row_ctr >= Istart) && (row_ctr < Iend)) {
          // construct B-part only if in the good range ...
          std::vector<unsigned short int> confB =
              basis_pointer->Confs_in_B[nsa][cb];
          for (int r = 0; r < LB; ++r) {
            config[r + LA] = confB[r];
          }
          // now do Hamiltonian basis size ...
          std::vector<unsigned short int> newconfig = config;
          std::vector<unsigned short int> newconfA = confA;
          std::vector<unsigned short int> newconfB = confB;
          int j;
          double diag = 0.;
          for (int r = 0; r < L; ++r) {
            if (coupling[r]) {
              int rb = r;
              int rb2 = (r + 1) % L;
              if (config[rb] == config[rb2]) {
                // Only diagonal term
                diag += 0.25 * Delta * coupling[r];
              } else {
                // diagonal term
                diag -= 0.25 * Delta * coupling[r];
                
                // off-diagonal term
                newconfig[rb] = 1 - config[rb];
                newconfig[rb2] = 1 - config[rb2];

                int nleft = 0;
                int nright = 0;
                for (int p = 0; p < LA; ++p) {
                  newconfA[p] = newconfig[p];
                  nleft += newconfig[p];
                }
                for (int p = 0; p < LB; ++p) {
                  newconfB[p] = newconfig[p + LA];
                  nright += newconfig[p + LA];
                }

                int new_nsa = basis_pointer->particle_sector[std::make_pair(
                    nleft,
                    nright)];  // int
                               // new_nsb=basis_pointer->partner_sector[new_nsa];
                newconfig[rb] = config[rb];
                newconfig[rb2] = config[rb2];
                j = basis_pointer->starting_conf[new_nsa] +
                    basis_pointer->InverseMapA[new_nsa][newconfA] *
                        basis_pointer->Confs_in_B[new_nsa].size() +
                    basis_pointer->InverseMapB[new_nsa][newconfB];
                MatSetValue(*pointer_to_H, row_ctr, j,
                            (PetscScalar)+0.5 * coupling[r], ADD_VALUES);
              }
            }
            // field part
            if (config[r]) {
              diag -= field[r] * 0.5;
            } else {
              diag += field[r] * 0.5;
            }
          }
          diag += bigshift;
          if (comparison_matrix) { diag=abs(diag);}
          if (diag != 0) {
            MatSetValue(*pointer_to_H, row_ctr, row_ctr, (PetscScalar)diag,
                        ADD_VALUES);
          }
        }
        row_ctr++;
      }  // loop over cb
    }    // over ca
  }      // over nsA sectors
}
/*
std::vector<unsigned long int> Hamiltonian::get_neighbors(
    unsigned long int index) {
  int i = 0;
  std::vector<unsigned long int> nn(0);
  std::vector<unsigned short int> config(L, 0);
  std::vector<unsigned short int> newconfig(L, 0);
  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        if (i == index) {
          std::vector<unsigned short int> confA =
              basis_pointer->Confs_in_A[nsa][ca];

          for (int r = 0; r < LA; ++r) {
            config[r] = confA[r];
          }
          std::vector<unsigned short int> confB =
              basis_pointer->Confs_in_B[nsa][cb];
          for (int r = 0; r < LB; ++r) {
            config[r + LA] = confB[r];
          }
          std::vector<unsigned short int> newconfig = config;
          std::vector<unsigned short int> newconfA = confA;
          std::vector<unsigned short int> newconfB = confB;
          for (int r = 0; r < L; ++r) {
            if (coupling[r]) {
              int rb = r;
              int rb2 = (r + 1) % L;
              if (config[rb] != config[rb2]) {
                newconfig[rb] = 1 - config[rb];
                newconfig[rb2] = 1 - config[rb2];

                int nleft = 0;
                int nright = 0;
                for (int p = 0; p < LA; ++p) {
                  newconfA[p] = newconfig[p];
                  nleft += newconfig[p];
                }
                for (int p = 0; p < LB; ++p) {
                  newconfB[p] = newconfig[p + LA];
                  nright += newconfig[p + LA];
                }

                int new_nsa = basis_pointer->particle_sector[std::make_pair(
                    nleft,
                    nright)];  // int
                               // new_nsb=basis_pointer->partner_sector[new_nsa];
                newconfig[rb] = config[rb];
                newconfig[rb2] = config[rb2];
                nn.push_back(basis_pointer->starting_conf[new_nsa] +
                             basis_pointer->InverseMapA[new_nsa][newconfA] *
                                 basis_pointer->Confs_in_B[new_nsa].size() +
                             basis_pointer->InverseMapB[new_nsa][newconfB]);
              }
            }
          }
          return nn;
        }
        i++;
      }
    }
  }
}
*/
void Hamiltonian::create_matrix_lapack(double *A) {
  unsigned long long int nconf = basis_pointer->total_number_of_confs;

  int row_ctr = 0;
  int j = 0;
  for (int nsa = 0; nsa < basis_pointer->valid_sectors; ++nsa) {
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      std::vector<unsigned short int> config(L, 0);
      std::vector<unsigned short int> confA =
          basis_pointer->Confs_in_A[nsa][ca];
      for (int r = 0; r < LA; ++r) {
        config[r] = confA[r];
      }
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        std::vector<unsigned short int> confB =
            basis_pointer->Confs_in_B[nsa][cb];
        for (int r = 0; r < LB; ++r) {
          config[r + LA] = confB[r];
        }

        // now do Hamiltonian basis size ...
        std::vector<unsigned short int> newconfig = config;
        std::vector<unsigned short int> newconfA = confA;
        std::vector<unsigned short int> newconfB = confB;
        int j;
        double diag = 0.;
        for (int r = 0; r < L; ++r) {
          if (coupling[r]) {
            int rb = r;
            int rb2 = (r + 1) % L;
            if (config[rb] == config[rb2]) {
              diag += 0.25 * Delta * coupling[r];
            } else {
              diag -= 0.25 * Delta * coupling[r];
              newconfig[rb] = 1 - config[rb];
              newconfig[rb2] = 1 - config[rb2];

              int nleft = 0;
              int nright = 0;
              for (int p = 0; p < LA; ++p) {
                newconfA[p] = newconfig[p];
                nleft += newconfig[p];
              }
              for (int p = 0; p < LB; ++p) {
                newconfB[p] = newconfig[p + LA];
                nright += newconfig[p + LA];
              }

              int new_nsa = basis_pointer->particle_sector[std::make_pair(
                  nleft,
                  nright)];  // int
                             // new_nsb=basis_pointer->partner_sector[new_nsa];
              newconfig[rb] = config[rb];
              newconfig[rb2] = config[rb2];
              j = basis_pointer->starting_conf[new_nsa] +
                  basis_pointer->InverseMapA[new_nsa][newconfA] *
                      basis_pointer->Confs_in_B[new_nsa].size() +
                  basis_pointer->InverseMapB[new_nsa][newconfB];
              A[j * nconf + row_ctr] += 0.5 * coupling[r];
            }
          }
          // field part
          if (config[r]) {
            diag -= field[r] * 0.5;
          } else {
            diag += field[r] * 0.5;
          }
        }
        diag += bigshift;
        if (diag != 0) {
          A[row_ctr * nconf + row_ctr] += diag;
        }
        row_ctr++;
      }  // loop over cb c
    }    // over ca
  }      // over nsA sectors
}

void Hamiltonian::diagonalize_lapack(double *A, double w[], bool eigenvectors) {
  MKL_INT myn = basis_pointer->total_number_of_confs;
  // double w[myn];
  // Real diag
  MKL_INT lda = myn, info, liwork;
  MKL_INT lwork;
  MKL_INT iwkopt;
  MKL_INT *iwork;
  // Real diag
  double wkopt;
  double *work;
  /* Query the optimal workspace */
  lwork = -1;
  liwork = -1;
#ifdef USE_MKL
  if (eigenvectors) {dsyevd("V", "Lower", &myn, &A[0], &lda, w, &wkopt, &lwork, &iwkopt,&liwork, &info);}
  else { dsyevd("N", "Lower", &myn, &A[0], &lda, w, &wkopt, &lwork, &iwkopt,&liwork, &info);}
#else
// LAPACK_dsyevd(LAPACK_ROW_MAJOR,"N", "Lower", (lapack_int) myn, &A[0], (lapack_int) lda, w);

if (eigenvectors) { LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V', 'L', (int) myn,&A[0], (int) myn, w);}
else { LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'N', 'L', (int) myn,&A[0], (int) myn, w);}

//dsyevd_(vectors, "Lower", &myn, &A[0], &lda, w, &wkopt, &lwork, &iwkopt,
//         &liwork, &info);
#endif
  // Allocate the optimal workspace
  lwork = (MKL_INT)wkopt;
  work = (double *)malloc(lwork * sizeof(double));
  liwork = iwkopt;
  iwork = (MKL_INT *)malloc(liwork * sizeof(MKL_INT));
  // Perform the diagonalization
#ifdef USE_MKL
  if (eigenvectors) { dsyevd("V", "Lower", &myn, &A[0], &lda, w, work, &lwork, iwork, &liwork,
         &info);}
         else { dsyevd("N", "Lower", &myn, &A[0], &lda, w, work, &lwork, iwork, &liwork,
         &info);}
#else
//dsyevd_(vectors, "Lower", &myn, &A[0], &lda, w, &wkopt, &lwork, &iwkopt,
//         &liwork, &info);
#endif
  free((void *)iwork);
  free((void *)work);
  // return w;
}

void Hamiltonian::print_local_fields(){
    ofstream fieldout;
    std::stringstream filename;
    filename << "LocalField." << basis_pointer->string_names << string_names << ".dat";
    fieldout.open(filename.str().c_str());
    fieldout.precision(16);

    for (int i = 0; i < L; i++) {
      fieldout << field[i] << "\n";
    }
    fieldout.close();
}



#endif
