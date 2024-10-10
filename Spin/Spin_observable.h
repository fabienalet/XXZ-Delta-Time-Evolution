#ifndef OBS_H
#define OBS_H


class observable {
  typedef std::vector<unsigned short int> Conf;

 private:
  basis *basis_pointer;

 public:
  observable(basis *this_basis_pointer, int this_number_threads);
  observable() {}
  ~observable() {}

  int number_threads;
  int nconf;
  double part_entropy(PetscScalar *state, double q);
  std::vector<double> all_part_entropy(PetscScalar *state, int Nq, double qmin,
                                       double qmax);
  std::vector<double> eigenvector_largest_coefficients(PetscScalar *state,
                                                       int Ncoeffs, int option);

  double Smax(PetscScalar *state);

  double KLd(PetscScalar *state1, PetscScalar *state2);
  double entang_entropy(double q);
  void compute_entanglement_spectrum(PetscScalar *state);

  void compute_special_matrix_brute_force(PetscScalar *state1, PetscScalar *state2,double scalar_product);
  double return_probability(PetscScalar *state, int i0) {
    double a = PetscAbsScalar(state[i0]);
    return (a * a);
  }
  std::vector<PetscScalar> return_correlations(PetscScalar *state1,
                                               PetscScalar *state2);
  void compute_local_magnetization(PetscScalar *state);
  void compute_local_magnetization_sandwich(PetscScalar *state1,PetscScalar *state2 );

  void compute_two_points_correlation(PetscScalar *state);
  std::vector<std::vector<double>> get_two_points_connected_correlation(
      PetscScalar *state);
  std::vector<std::vector<double>> get_SpSm_correlation(PetscScalar *state);
  
  std::vector<std::vector<double>> get_fermionic_correlation(
      PetscScalar *state);
  std::vector<double> sz_local;
  std::vector<double> sz_local_sandwich;
  // two points correlation function
  std::vector<std::vector<double>> G;
  double product_state_imbalance(int, int, int);

  std::vector<double> entanglement_spectrum;
  void init();
  //	std::vector<int_type> Reshape;

  std::vector<double> sz_for_basis_state;
};

bool compare_squared(double a, double b) { return (a * a < b * b); }

std::vector<double> observable::eigenvector_largest_coefficients(
    PetscScalar *state, int Ncoeffs, int option) {
  int n = nconf;
  std::vector<double> largest_coeffs(Ncoeffs, 0.);

  for (int i = 0; i < n; i++) {
    // The vector largest_coeffs contains Ncoeffs ai values already encountered
    // The first index contains the highest ai value already seen
    // It is always sorted by decreasing ai
     double ai = PetscAbsScalar(state[i]);

//#ifdef PETSC_USE_COMPLEX

    //double ai = PetscAbsComplex(state[i]);
    if (pow(largest_coeffs[Ncoeffs - 1], 2) < pow(ai, 2)) {
      largest_coeffs.pop_back();
      largest_coeffs.push_back(ai);
      std::sort(largest_coeffs.rbegin(), largest_coeffs.rend(),
                compare_squared);
    }
  }

  // We kept |ai| to keep it away from the machine precision range
  // Now we square it to obtain the probability
  if (option == 0) {
    for (int i = 0; i < Ncoeffs; ++i) {
      largest_coeffs[i] *= largest_coeffs[i];
    }
  }

  return largest_coeffs;
}

observable::observable(basis *this_basis_pointer, int this_number_threads) {
  number_threads = this_number_threads;
  basis_pointer = this_basis_pointer;
  nconf = basis_pointer->total_number_of_confs;

  sz_for_basis_state.resize(basis_pointer->NUMBER_OF_STATES, 0);

  sz_for_basis_state = basis_pointer->sz_for_basis_state;
}

std::vector<PetscScalar> observable::return_correlations(PetscScalar *state1,
                                                         PetscScalar *state2) {
  //  cout << "Entering correl : " << basis_pointer->L << " " <<
  //  basis_pointer->LA << endl; std::cout << "Cra1\n";
  int L = basis_pointer->L;
  int LA = basis_pointer->LA;
  std::vector<PetscScalar> correl(L, 0.);
  int i = 0;

  // std::cout << L << " " << LA << "Cra2\n";

  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        for (int d = 0; d < L; ++d) {
          short int prefactor = 0;

          if (d < LA) {
            prefactor =
                sz_for_basis_state[basis_pointer->Confs_in_A[nsa][ca][d]];
          }
          // switch (basis_pointer->Confs_in_A[nsa][ca][d]) { case 1 :
          // prefactor=1; break; case 2 : prefactor=1; break; case 3 :
          // prefactor=2; break;} }
          else {
            prefactor =
                sz_for_basis_state[basis_pointer->Confs_in_B[nsa][cb][d - LA]];
          }
          //	   cout << "Testing d=" << d << " pref=" << prefactor <<
          // std::endl; switch (basis_pointer->Confs_in_A[nsb][cb][d-LA]) { case
          // 1 : prefactor=1; break; case 2 : prefactor=1; break; case 3 :
          // prefactor=2; break;} }
#ifdef PETSC_USE_COMPLEX
          if (prefactor != 0) {
            correl[d] += prefactor * state1[i] * PetscConjComplex(state2[i]);
          }
#else
          if (prefactor != 0) {
            correl[d] += prefactor * state1[i] * state2[i];
          }
#endif
        }
        i++;
      }
    }
  }

  return correl;
}

void observable::compute_local_magnetization(PetscScalar *state) {
  /*
    Compute <state|S_i^z|state> for all sites i
  */
  int L = basis_pointer->L;
  int LA = basis_pointer->LA;
  // local magnetization
  sz_local.resize(L, 0.);
  for (int i = 0; i < L; i++) sz_local[i] = 0.;
  std::vector<int> local_state(L, 0);

  int i = 0;  // configuration index
  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        for (int d = 0; d < L; ++d) {
          if (d < LA) {
            local_state[d] = basis_pointer->Confs_in_A[nsa][ca][d];
          } else {
            local_state[d] = basis_pointer->Confs_in_B[nsa][cb][d - LA];
          }
// sz_for_basis_state[local_state[d]] = {Sz at site d in the current
// configuration}
#ifdef PETSC_USE_COMPLEX
          sz_local[d] += sz_for_basis_state[local_state[d]] *
                         PetscRealPart(state[i] * PetscConjComplex(state[i]));
#else
          sz_local[d] +=
              sz_for_basis_state[local_state[d]] * state[i] * state[i];
#endif
        }
        i++;
      }
    }
  }
}

void observable::compute_local_magnetization_sandwich(PetscScalar *state1,PetscScalar *state2 ) {
  /*
    Compute <state|S_i^z|state> for all sites i
  */
  int L = basis_pointer->L;
  int LA = basis_pointer->LA;
  // local magnetization
  sz_local_sandwich.resize(L, 0.);
  for (int i = 0; i < L; i++) sz_local_sandwich[i] = 0.;
  std::vector<int> local_state(L, 0);

  int i = 0;  // configuration index
  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        for (int d = 0; d < L; ++d) {
          if (d < LA) {
            local_state[d] = basis_pointer->Confs_in_A[nsa][ca][d];
          } else {
            local_state[d] = basis_pointer->Confs_in_B[nsa][cb][d - LA];
          }
// sz_for_basis_state[local_state[d]] = {Sz at site d in the current
// configuration}
#ifdef PETSC_USE_COMPLEX
          sz_local_sandwich[d] += sz_for_basis_state[local_state[d]] *
                         PetscRealPart(state1[i] * PetscConjComplex(state2[i]));
#else
          sz_local_sandwich[d] +=
              sz_for_basis_state[local_state[d]] * state1[i] * state2[i];
#endif
        }
        i++;
      }
    }
  }
}

void observable::compute_two_points_correlation(PetscScalar *state) {
  /*
  Compute <state|S_i^zS_j^z|state>
  */
  int L = basis_pointer->L;
  int LA = basis_pointer->LA;
  // two points correlation function
  G.resize(L, std::vector<double>(L, 0.));
  for (int i = 0; i < L; i++)
    for (int j = 0; j < L; j++) G[i][j] = 0.;

  int c = 0;  // configuration index
  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        for (int i = 0; i < L; ++i) {
          int statei;
          if (i < LA)
            statei = basis_pointer->Confs_in_A[nsa][ca][i];
          else
            statei = basis_pointer->Confs_in_B[nsa][cb][i - LA];
          for (int j = i; j < L; ++j) {
            int statej;
            if (j < LA)
              statej = basis_pointer->Confs_in_A[nsa][ca][j];
            else
              statej = basis_pointer->Confs_in_B[nsa][cb][j - LA];
            // sz_for_basis_state[local_state[d]] = {Sz at site d in the current
            // configuration}
            double SizSjz =
                sz_for_basis_state[statei] * sz_for_basis_state[statej];
#ifdef PETSC_USE_COMPLEX
            G[i][j] +=
                SizSjz * PetscRealPart(state[c] * PetscConjComplex(state[c]));
#else
            G[i][j] += SizSjz * state[c] * state[c];
#endif
          }
        }
        c++;
      }
    }
  }
  // the matrix is symmetric
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < i; j++) {
      G[i][j] = G[j][i];
    }
  }
}

std::vector<std::vector<double>> observable::get_SpSm_correlation(
    PetscScalar *state) {
  /* Return <state|S_i^+ S_j^-|state> */
  // currently this works only for spin-1/2
  if (basis_pointer->NUMBER_OF_STATES != 2) {
    std::cout << "Incorrect number of spin states "
              << basis_pointer->NUMBER_OF_STATES << std::endl;
  }
  int L = basis_pointer->L;
  int LA = basis_pointer->LA;
  std::vector<std::vector<double>> corr(L, std::vector<double>(L));

  int c = 0;  // configuration index

  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {

      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {

        for (int i = 0; i < L; ++i) {
          int statei;
          if (i < LA)
            statei = basis_pointer->Confs_in_A[nsa][ca][i];
          else
            statei = basis_pointer->Confs_in_B[nsa][cb][i - LA];
          
          for (int j = i; j < L; ++j) {
            int statej;
            if (j < LA)
              statej = basis_pointer->Confs_in_A[nsa][ca][j];
            else
              statej = basis_pointer->Confs_in_B[nsa][cb][j - LA];

            if ((sz_for_basis_state[statej] == 0.5) and
                (sz_for_basis_state[statei] == -0.5)) {

              // conf with spins at sites i and j reversed
              Conf Cra = basis_pointer->Confs_in_A[nsa][ca];
              Conf Crb = basis_pointer->Confs_in_B[nsa][cb];
              // reverse spins at sites i and j
              if (i < LA)
                Cra[i] = (Cra[i] + 1) % 2;
              else
                Crb[i - LA] = (Crb[i - LA] + 1) % 2;
              
              if (j < LA)
                Cra[j] = (Cra[j] + 1) % 2;
              else
                Crb[j - LA] = (Crb[j - LA] + 1) % 2;

              int nleft = 0;
              for (int statea : Cra) // changed state to statea
                nleft += basis_pointer->numberparticle_for_basis_state[statea];

              int nright = 0;
              for (int stateb : Crb)// changed state to stateb
                nright += basis_pointer->numberparticle_for_basis_state[stateb];

              int nsra =
                  basis_pointer->particle_sector[std::make_pair(nleft, nright)];

              int cra = basis_pointer->InverseMapA[nsra][Cra];
              int crb = basis_pointer->InverseMapB[nsra][Crb];

              // compute the index of the reversed conf
              int cr = basis_pointer->starting_conf[nsra] +
                       cra * basis_pointer->Confs_in_B[nsra].size() + crb;

#ifdef PETSC_USE_COMPLEX
              corr[i][j] +=
                  PetscRealPart(state[c] * PetscConjComplex(state[cr]));
#else
              corr[i][j] += state[c] * state[cr];
#endif
            } else if ((i == j) and (sz_for_basis_state[statej] == 0.5)) {
#ifdef PETSC_USE_COMPLEX
              corr[i][j] +=
                  PetscRealPart(state[c] * PetscConjComplex(state[c]));
#else
              corr[i][j] += state[c] * state[c];
#endif
            }
          }
        }

        c++;

      }
    }
  }
  // the matrix is symmetric
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < i; j++) {
      corr[i][j] = corr[j][i];
    }
  }
  return corr;
}

std::vector<std::vector<double>> observable::get_fermionic_correlation(
    PetscScalar *state) {
  /*
  Return <state|c_i^+ c_j^-|state>
  Rk: here we assume "one fermion" corresponds to down spin, and "no fermion" to
  up spin. Hence the correlation is <S_i^- (-1)^#(down spins in [i,j)) S_j^+>
  */
  // currently this works only for spin-1/2
  if (basis_pointer->NUMBER_OF_STATES != 2) {
    std::cout << "Incorrect number of spin states "
              << basis_pointer->NUMBER_OF_STATES << std::endl;
  }
  int L = basis_pointer->L;
  int LA = basis_pointer->LA;
  std::vector<std::vector<double>> corr(L, std::vector<double>(L));
  int c = 0;  // configuration index
  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (int ca = 0; ca < basis_pointer->Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < basis_pointer->Confs_in_B[nsa].size(); ++cb) {
        for (int i = 0; i < L; ++i) {
          int statei;
          if (i < LA)
            statei = basis_pointer->Confs_in_A[nsa][ca][i];
          else
            statei = basis_pointer->Confs_in_B[nsa][cb][i - LA];
          for (int j = i; j < L; ++j) {
            int statej;
            if (j < LA)
              statej = basis_pointer->Confs_in_A[nsa][ca][j];
            else
              statej = basis_pointer->Confs_in_B[nsa][cb][j - LA];
            if ((sz_for_basis_state[statej] == -0.5) and
                (sz_for_basis_state[statei] == 0.5)) {
              // concatenate the left and right parts of the current
              // configuration
              Conf conc_conf = basis_pointer->Confs_in_A[nsa][ca];
              conc_conf.insert(conc_conf.end(),
                               basis_pointer->Confs_in_B[nsa][cb].begin(),
                               basis_pointer->Confs_in_B[nsa][cb].end());
              // count the number of down spins (or fermions) in the interval
              // [i,j) in the current configuration
              int ndown = 0;
              for (int k = i; k < j; k++) ndown += 1 - conc_conf[k];
              // +1 if this number is even
              int sign = -2 * (ndown % 2) + 1;

              // conf with spins at sites i and j reversed
              Conf Cra = basis_pointer->Confs_in_A[nsa][ca];
              Conf Crb = basis_pointer->Confs_in_B[nsa][cb];
              // reverse spins at sites i and j
              if (i < LA)
                Cra[i] = (Cra[i] + 1) % 2;
              else
                Crb[i - LA] = (Crb[i - LA] + 1) % 2;
              if (j < LA)
                Cra[j] = (Cra[j] + 1) % 2;
              else
                Crb[j - LA] = (Crb[j - LA] + 1) % 2;

              int nleft = 0;
              for (int state : Cra)
                nleft += basis_pointer->numberparticle_for_basis_state[state];
              int nright = 0;
              for (int state : Crb)
                nright += basis_pointer->numberparticle_for_basis_state[state];
              int nsra =
                  basis_pointer->particle_sector[std::make_pair(nleft, nright)];
              int cra = basis_pointer->InverseMapA[nsra][Cra];
              int crb = basis_pointer->InverseMapB[nsra][Crb];
              // compute the index of the reversed conf
              int cr = basis_pointer->starting_conf[nsra] +
                       cra * basis_pointer->Confs_in_B[nsra].size() + crb;

#ifdef PETSC_USE_COMPLEX
              corr[i][j] +=
                  sign * PetscRealPart(state[c] * PetscConjComplex(state[cr]));
#else
              corr[i][j] += sign * state[c] * state[cr];
#endif
            } else if ((i == j) and (sz_for_basis_state[statej] == -0.5)) {
#ifdef PETSC_USE_COMPLEX
              corr[i][j] +=
                  PetscRealPart(state[c] * PetscConjComplex(state[c]));
#else
              corr[i][j] += state[c] * state[c];
#endif
            }
          }
        }
        c++;
      }
    }
  }
  // the matrix is symmetric
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < i; j++) {
      corr[i][j] = corr[j][i];
    }
  }
  return corr;
}

std::vector<std::vector<double>>
observable::get_two_points_connected_correlation(PetscScalar *state) {
  compute_local_magnetization(state);
  compute_two_points_correlation(state);
  int L = basis_pointer->L;
  std::vector<std::vector<double>> Gc(L, std::vector<double>(L));
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < L; j++) {
      Gc[i][j] = G[i][j] - sz_local[i] * sz_local[j];
    }
  }
  return Gc;
}

double observable::product_state_imbalance(int nsa0, int ca0, int cb0) {
  /*
  Return the observable <S_i^z(t)S_i^z(0)> assuming initial state is the product
  state index0, corresponding to the configuration (nsa0, ca0, cb0) in the
  computational basis. Local magnetization for the time-evolved state has to be
  called before this.
  */
  double imbalance = 0;
  int L = basis_pointer->L;
  int LA = basis_pointer->LA;
  imbalance = 0;
  std::vector<int> local_state(L, 0);
  for (int d = 0; d < L; ++d) {
    if (d < LA) {
      local_state[d] = basis_pointer->Confs_in_A[nsa0][ca0][d];
    } else {
      local_state[d] = basis_pointer->Confs_in_B[nsa0][cb0][d - LA];
    }
    imbalance += sz_for_basis_state[local_state[d]] * sz_local[d];
  }
  imbalance /= (L / 4.);
  return imbalance;
}

double observable::part_entropy(PetscScalar *state, double q) {
  int n = nconf;
  double entropy = 0;
  if (q == 1) {
    for (int i = 0; i < n; i++) {
      double ai = PetscAbsScalar(state[i]);
      if (ai != 0) {
        entropy += -2.0 * ai * ai * log(ai);
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      double ai = PetscAbsScalar(state[i]);
      entropy += pow(ai * ai, q);
    }
    entropy = log(entropy) / (1. - q);
  }
  return entropy;
}

std::vector<double> observable::all_part_entropy(PetscScalar *state, int Nq,
                                                 double qmin, double qmax) {
  int n = nconf;

  std::vector<double> qs;
  double mymin = qmin;
  double current_q;
  if (qmin < 1) {
    current_q = qmin;
    for (int p = 0; p < 9; ++p) {
      qs.push_back(current_q);
      current_q += (1. - qmin) / 9.;
    }
    mymin = 1;
  }
  current_q = mymin;
  double dq = (qmax - mymin) / (Nq - 1);
  for (int nq = 0; nq < Nq - 1; ++nq) {
    qs.push_back(current_q);
    current_q += dq;
  }
  qs.push_back(current_q);
  double real_Nq = qs.size();
  std::vector<double> entropy(real_Nq);
  for (int i = 0; i < n; i++) {
    double ai = PetscAbsScalar(state[i]);
    if (ai != 0) {
      for (int nq = 0; nq < qs.size(); ++nq) {
        double q = qs[nq];
        if (q == 1) {
          entropy[nq] += -2.0 * ai * ai * log(ai);
        } else {
          entropy[nq] += pow(ai * ai, q);
        }
      }
    }
  }

  double q = qmin;
  for (int nq = 0; nq < qs.size(); ++nq) {
    double q = qs[nq];
    if (q != 1) {
      entropy[nq] = log(entropy[nq]) / (1. - q);
    }
  }

  return entropy;
}

double observable::Smax(PetscScalar *state) {
  int n = nconf;
  double amax = 0;
  for (int i = 0; i < n; i++) {
    double ai = PetscAbsScalar(state[i]);
    if (ai > amax) {
      amax = ai;
    }
  }
  return -log(amax * amax);
}

double observable::KLd(PetscScalar *state1, PetscScalar *state2) {
  int n = nconf;
  double entropy = 0;
  for (int i = 0; i < n; i++) {
    double ai = PetscAbsScalar(state1[i]);
    double bi = PetscAbsScalar(state2[i]);
    if ((ai != 0) && (bi != 0)) {
      entropy += -2.0 * ai * ai * log(ai / bi);
    }
  }
  return entropy;
}

double observable::entang_entropy(double q) {
  int n = entanglement_spectrum.size();
  double entropy = 0.;

  if (q == 1) {
    for (int i = 0; i < n; i++) {
      double ai = entanglement_spectrum[i];
      if (ai != 0) {
        entropy += -ai * log(ai);
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      double ai = entanglement_spectrum[i];
      entropy += pow(ai, q);
    }
    entropy = log(entropy) / (1. - q);
  }
  return entropy;
}

void observable::compute_entanglement_spectrum(PetscScalar *state) {
  entanglement_spectrum.resize(0);
// cout << "*** State \n";
// for (int p=0;p<number_valid_coverings;++p) {
//	cout << state[p] << endl;
//}
// cout << "Starting entanglement with MKL=" << mkl_get_max_threads() << endl;
#ifdef USE_MKL
  mkl_set_num_threads(number_threads);
#else
  omp_set_num_threads(number_threads);
#endif

  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    int sizeA = basis_pointer->Confs_in_A[nsa].size();
    int sizeB = basis_pointer->Confs_in_B[nsa].size();
    int sectorsize = sizeA * sizeB;
    int start = basis_pointer->starting_conf[nsa];
    //	cout << "Sector " << nsa << " " << sizeA << " " << sizeB << " " << start
    //<< std::endl;

    if ((sectorsize > 0)) {
      if (sectorsize == 1) {
        int index = start;
        double ame;
        // if (index!=nconfs) { ame=PetscAbsScalar(state[index]);} else
        // {ame=0.;}
        ame = PetscAbsScalar(state[index]);
        entanglement_spectrum.push_back(ame * ame);
        //	cout << "Sector size=1; EE eigenvalue=" << ame*ame << endl;
      } else {  // sectorsize>1
#ifdef PETSC_USE_COMPLEX
        std::vector<MKL_Complex16> psi_reshaped;
#else
        std::vector<double> psi_reshaped;
#endif
        psi_reshaped.resize(sectorsize);
        int index;
        PetscScalar me;
        for (int p1 = 0; p1 < sectorsize; ++p1) {
          index = start + p1;
          //	if (index!=nconfs) { me=state[index];
          me = state[index];
#ifdef PETSC_USE_COMPLEX
          psi_reshaped[p1].real = PetscRealPart(me);
          psi_reshaped[p1].imag = PetscImaginaryPart(me);
#else
          psi_reshaped[p1] = me;
#endif
        }
        //}
        // now do svd
        int minsize = min(sizeA, sizeB);
        std::vector<double> local_svd_spectrum(minsize, 0.);
        // if (sizeA!=sizeB) std::cout << "** New sector A=" << sizeA << " B="
        // << sizeB << endl;
        {
          MKL_INT m = sizeB;
          MKL_INT n = sizeA;
          MKL_INT lda = m;
          MKL_INT ldu = m;
          MKL_INT ldvt = n;
          MKL_INT info, lwork;
          // if (sizeA!=sizeB) cout << "Selecting m=sizeA\n";
          MKL_INT iwork[8 * minsize];
          lwork = -1;
#ifdef PETSC_USE_COMPLEX
#ifdef USE_MKL
          MKL_Complex16 wkopt;
          MKL_Complex16 *work;
          MKL_Complex16 u[ldu * m], vt[ldvt * n];
          double rwork[5 * m * m + 7 * m];
          zgesdd("N", &m, &n, &psi_reshaped[0], &lda, &local_svd_spectrum[0], u,
                 &ldu, vt, &ldvt, &wkopt, &lwork, rwork, iwork, &info);
#else
          double __complex__ wkopt;
          double __complex__ *work;
          double __complex__ u[ldu * m], vt[ldvt * n];
          double rwork[5 * m * m + 7 * m];
        //  zgesdd_("N", &m, &n, (double __complex__ *)&psi_reshaped[0], &lda,&local_svd_spectrum[0], u, &ldu, vt, &ldvt, &wkopt, &lwork,rwork, iwork, &info);
        //          zgesdd_("N", &m, &n, (double __complex__ *)&psi_reshaped[0], &m,
       //           &local_svd_spectrum[0], u, &m, vt, &n, &wkopt, &lwork,
       //           rwork, iwork, &info);
          LAPACKE_zgesdd( LAPACK_ROW_MAJOR, "N", &m, &n, &psi_reshaped[0], &lda, &local_svd_spectrum[0], u,
                 &ldu, vt, &ldvt);
                 LAPACKE_zgesdd( LAPACK_ROW_MAJOR, "N", (lapack_int) m, (lapack_int)  n, (double __complex__ *) &psi_reshaped[0], (lapack_int)  lda, &local_svd_spectrum[0], u,
                 (lapack_int) ldu, vt, (lapack_int) ldvt);
#endif
#ifdef USE_MKL
          lwork = (MKL_INT)wkopt.real;
          work = (MKL_Complex16 *)malloc(lwork * sizeof(MKL_Complex16));
          zgesdd("N", &m, &n, &psi_reshaped[0], &lda, &local_svd_spectrum[0], u,
                 &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
#else
// TODO
//          lwork = (int) creal(wkopt);
 //         work = (double __complex__  *)malloc(lwork * sizeof(double __complex__ ));

//          zgesdd_("N", &m, &n, &psi_reshaped[0], &m, &local_svd_spectrum[0],
//                  u, &m, vt, &n, work, &lwork, rwork, iwork, &info);
#endif
#else
          double wkopt;
          double *work;
          double u[ldu * m], vt[ldvt * n];
#ifdef USE_MKL
          dgesdd("N", &m, &n, &psi_reshaped[0], &lda, &local_svd_spectrum[0], u,
                 &ldu, vt, &ldvt, &wkopt, &lwork, iwork, &info);
#else
          dgesdd_("N", &m, &n, &psi_reshaped[0], &lda, &local_svd_spectrum[0],
                  u, &ldu, vt, &ldvt, &wkopt, &lwork, iwork, &info);
#endif
          lwork = (MKL_INT)wkopt;
          work = (double *)malloc(lwork * sizeof(double));
#ifdef USE_MKL
          dgesdd("N", &m, &n, &psi_reshaped[0], &lda, &local_svd_spectrum[0], u,
                 &ldu, vt, &ldvt, work, &lwork, iwork, &info);
#else
          dgesdd_("N", &m, &n, &psi_reshaped[0], &lda, &local_svd_spectrum[0],
                  u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
#endif
#endif
          free((void *)work);
          //	mkl_set_num_threads(1);
        }

        double s;

        for (int rr = 0; rr < local_svd_spectrum.size(); ++rr) {
          double s = local_svd_spectrum[rr];
          entanglement_spectrum.push_back(s * s);
          //	  std::cout << s*s << endl;
        }

      }  // sectorsize>1
    }    // sectorsize>0
  }      // loop over sectors

  double sum = 0.;
  for (int pp = 0; pp < entanglement_spectrum.size(); ++pp) {
    sum += entanglement_spectrum[pp];
  }
#ifdef USE_MKL
  mkl_set_num_threads(1);
#else
  omp_set_num_threads(1);
#endif
}

void observable::compute_special_matrix_brute_force(PetscScalar *state1,PetscScalar *state2,double scalar_product) {
double cutoff_precision=1e-15;
PetscOptionsGetReal(NULL,NULL,"-cutoff_precision",&cutoff_precision,NULL);

PetscBool print=PETSC_FALSE;
PetscOptionsGetBool(NULL,NULL,"-special_print",&print,NULL);

  std::vector<double> special_spectrum;
  special_spectrum.resize(0);

#ifdef USE_MKL
  mkl_set_num_threads(number_threads);
#else
  omp_set_num_threads(number_threads);
#endif

  for (int nsa = 0; nsa < basis_pointer->valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    int sizeA = basis_pointer->Confs_in_A[nsa].size();
    int sizeB = basis_pointer->Confs_in_B[nsa].size();

    if ( (sizeA>0) && (sizeB>0) ) {
    int sectorsize = sizeA * sizeB;
    int start = basis_pointer->starting_conf[nsa];
    //	cout << "Sector " << nsa << " " << sizeA << " " << sizeB << " " << start
    //<< std::endl;

    int reduced_matrix_size_on_this_sector=sizeA*sizeA;
    std::vector<double> reducedA_this_sector(reduced_matrix_size_on_this_sector);

    for (int cai = 0; cai < sizeA; ++cai) {
      for (int caj = 0; caj < sizeA; ++caj) {
        int position_on_matrix=cai*sizeA+caj;//check

                double me=0;
            for (int cb = 0; cb < sizeB; ++cb) {
              me+=PetscRealPart(state1[start+sizeB*cai+cb])*PetscRealPart(state2[start+sizeB*caj+cb]);
            }
    reducedA_this_sector[position_on_matrix]=me/scalar_product;

  } }

  if (sectorsize == 1) {
  if (print)  cout << "Sector " << nsa << " of size " << sizeA << " , pushing  " << reducedA_this_sector[0] << endl;
    special_spectrum.push_back(reducedA_this_sector[0]);
  }
  else {
    if (print) {
	cout << "Sector " << nsa << " of size " << sizeA << " , the matrix is (Normalized) :\n[ ";
//  for (int k=0;k<reduced_matrix_size_on_this_sector;++k) { cout << reducedA_this_sector[k] << " ";}
  cout << "]\n";
}
  // Try a non-symmetric diagonalization
  std::vector<double> copyA=reducedA_this_sector;
  double tr=0.;
  for (int r=0;r<sizeA;++r) { tr+=copyA[r+sizeA*r];}

  MKL_INT n=sizeA;
  MKL_INT ld=sizeA;
  MKL_INT info;
  double scale[sizeA];
// #ifdef USE_MKL

  //std::cout << "Starting dgebal\n";
  MKL_INT ilo=1; MKL_INT ihi=sizeA;

//  dgebal("N", &sizeA, &copyA[0], &n, &ilo, &ihi, &scale[0],&info);
//  if (info != 0) cerr << "Error in LAPACKE_dgebal " << info << " error" << endl;
//  std::cout << "ilo=" << ilo << " ihi=" << ihi << endl;
  MKL_INT lwork=12*ld;
  // 12 seems to work (?) but not sure why
  double *work;
  work = (double *)malloc(lwork * sizeof(double));
  double *tau;
  tau = (double *)malloc(ld * sizeof(double));
  //std::vector<double> tau(sizeA-1);
//  std::cout << "Starting dgehrd1\n";

  dgehrd(&n, &ilo, &ihi, &copyA[0], &n, tau, work,&lwork,&info);
  /*
  lwork = (MKL_INT)work[0];
  work = (double *)malloc(lwork * sizeof(double));
  std::cout << "Starting dgehrd2\n";
  dgehrd(&n, &ilo, &ihi, &copyA[0], &n, &tau[0], work,&lwork,&info);
*/
    std::vector<double> WR(sizeA); std::vector<double> WI(sizeA);
    /*
    MKL_INT lwork2=-1;
    double *work2;
    std::cout << "Starting dhseqr1\n";
    dhseqr("E","N",&n,&ilo,&ihi,&copyA[0],&n,&WR[0],&WI[0],NULL,&n,work2,&lwork2,&info);
    lwork2 = (MKL_INT)work2[0];
    work2 = (double *)malloc(lwork2 * sizeof(double));
    */
//    std::cout << "Starting dhseqr2\n";


  //  dhseqr("E","N",&n,&ilo,&ihi,&copyA[0],&n,&WR[0],&WI[0],NULL,&n,work,&lwork,&info);
    if (info != 0) cerr << "Error in LAPACKE_dhseqr " << info << " error" << endl;
  if (print)  std::cout << "**** eigenvalues seems to be \n";

  free((void *)work);
  free((void *)tau);

    double trr=0; double tri=0.;
  for (int r=0;r<sizeA;++r) {

    if (print)  std::cout << WR[r] << " +i " << WI[r] << endl;
      special_spectrum.push_back(WR[r]);
      trr+=WR[r]; tri+=WI[r];
      if ( (WR[r]<-cutoff_precision) || (fabs(WI[r])>cutoff_precision) ) {
        std::cout << "NEGATIVE OR COMPLEX : Sector " << nsa << " of size " << sizeA << ", eigenvalue= " << r << " : " << WR[r] << " +i " << WI[r] << endl;
}
if ( (fabs(WR[r])<cutoff_precision) ) {
  //std::cout << "ZERO : Sector " << nsa << " of size " << sizeA << ", eigenvalue= " << r << " : " << WR[r] << " +i " << WI[r] << endl;
}

       }
if (print)    std::cout << "Trace should be Tr=" << tr << " and it is " << trr << " +i " << tri << endl;
  } // sectorsize!=1
} // sectorsize >0



}
double S1=0.; double S2=0.;
int q=2;
//double cutoff_precision=1e-15;
for (int k=0;k<special_spectrum.size();++k)
{
      double ai = special_spectrum[k];
      if (ai >= cutoff_precision) {
        S1 += -ai * log(ai);
      }
      S2+= pow(ai, q);

  }
  S2 = log(S2) / (1. - q);
  cout << "S1= " << S1 << " S2= " << S2 << endl;



#ifdef USE_MKL
  mkl_set_num_threads(1);
#else
  omp_set_num_threads(1);
#endif
}

#endif
