#ifndef BASIS_H
#define BASIS_H

class basis {
  typedef std::vector<unsigned short int> Conf;

 private:
 public:
  basis(int L_, int sector, int myrank_, int num_states_);
  basis(int L_, int LA_, int sector, int myrank_, int num_states_);
  basis(){};
  ~basis() {}

  void change_basis(const basis& b0, const PetscScalar* state,
                    PetscScalar* new_state);

  short int NUMBER_OF_STATES;
  short int Q_;


  std::vector<int> numberparticle_for_basis_state;

  void int_to_bosons(unsigned long int num, int powof2, int L, Conf& bosons);
  bool is_config_valid_inA(int num, int powof2, int L, int Nmax);

  std::vector<unsigned int> number_conf_in_A;
  std::vector<std::vector<Conf> > Confs_in_A;
  std::vector<unsigned int> starting_conf;
  std::vector<std::map<Conf, unsigned long int> > InverseMapA;

  std::vector<unsigned int> number_conf_in_B;
  std::vector<std::vector<Conf> > Confs_in_B;
  std::vector<std::map<Conf, unsigned long int> > InverseMapB;

  int ZN_Sector;
  bool use_parity;
  int Nparticles;
  std::map<std::pair<int, int>, int> particle_sector;
  int valid_sectors;
  int LA, LB;
  int L;
  int myrank;
  int total_number_of_confs;

  void clean_basis();
  void init();
  unsigned long int index(Conf conf);
  void init_vectors_product_state(Vec& Psi_t, Vec& Psi_t2, int i0);
  void init_vectors_random(Vec& Psi_t, Vec& Psi_t2, int seed3, PetscInt Istart,
                           PetscInt Iend);
  void init_vector_random(Vec& Psi_t, int seed3, PetscInt Istart,
                          PetscInt Iend);
  void get_conf_coefficients(unsigned long int index, int& nsa, int& ca,
                             int& cb);

  std::string string_names;
};

basis::basis(int L_, int ZN_Sector_, int myrank_, int num_states_) {
  // size of the Hilbert space of 1 spin
  Q_=num_states_;
  NUMBER_OF_STATES = num_states_;
  L = L_;
  myrank = myrank_;
  ZN_Sector = ZN_Sector_;

  LA = L / 2;
  LB = L - LA;
  init();
}

basis::basis(int L_, int LA_, int ZN_Sector_, int myrank_, int num_states_) {
  // size of the Hilbert space of 1 spin
  NUMBER_OF_STATES = num_states_;
  Q_=num_states_;
  L = L_;
  myrank = myrank_;
  ZN_Sector = ZN_Sector_;

  LA = LA_;
  LB = L - LA;
  init();
}

void basis::clean_basis() {
  number_conf_in_A.clear();
  Confs_in_A.clear();
  starting_conf.clear();
  InverseMapA.clear();
  number_conf_in_B.clear();
  Confs_in_B.clear();
  // starting_confB.clear();
  InverseMapB.clear();

  particle_sector.clear();
  // partner_sector.clear();
}

void basis::init() {

  numberparticle_for_basis_state.resize(NUMBER_OF_STATES, 0);
  for (int r = 0; r < NUMBER_OF_STATES; ++r) {
    numberparticle_for_basis_state[r] = r;
  }
  valid_sectors = 0;
//std::cout << "asking for total parity = " << ZN_Sector << std::endl;
  for (int parity_left = 0; parity_left < Q_; ++parity_left) {
    int parity_right = (ZN_Sector + Q_ - parity_left)%Q_;
    //if ((nright >= 0) && (nright <= (LB * NUMBER_OF_STATES))) {
      if (1) {
      particle_sector[std::make_pair(parity_left, parity_right)] = valid_sectors;
    //  std::cout << "Sector " << valid_sectors << " is " << parity_left << " " << parity_right << std::endl;
      valid_sectors++;
    }
  }

  number_conf_in_A.resize(valid_sectors);
  starting_conf.resize(valid_sectors + 1);
  InverseMapA.resize(valid_sectors);
  Confs_in_A.resize(valid_sectors);

  number_conf_in_B.resize(
      valid_sectors);  // starting_confB.resize(valid_sectors+1);
  InverseMapB.resize(valid_sectors);
  Confs_in_B.resize(valid_sectors);

  int correct_power_of_two = 1;
  while (pow(2, correct_power_of_two) < NUMBER_OF_STATES) {
    correct_power_of_two++;
  }

  for (unsigned long int r = 0; r < pow(pow(2, correct_power_of_two), LA);
       ++r) {
    if (is_config_valid_inA(r, correct_power_of_two, LA,
                            LA * NUMBER_OF_STATES)) {
  //  if (1) {
      std::vector<unsigned short int> this_conf(LA, 0);
      int_to_bosons(r, correct_power_of_two, LA, this_conf);
      int nleft = 0;
      for (int p = 0; p < LA; ++p) {
        nleft += numberparticle_for_basis_state[this_conf[p]];
      }
      int parity_left=nleft%Q_;

      int parity_right = (ZN_Sector + Q_ - parity_left)%Q_;
      std::map<std::pair<int, int>, int>::const_iterator it =
          particle_sector.find(std::make_pair(parity_left, parity_right));
      if (it != particle_sector.end()) {
        if ((InverseMapA[it->second]).count(this_conf) == 0) {
          Confs_in_A[it->second].push_back(this_conf);
          InverseMapA[it->second][this_conf] = number_conf_in_A[it->second];
          number_conf_in_A[it->second] += 1;
        }
      }
    }
  }

  for (unsigned long int r = 0; r < pow(pow(2, correct_power_of_two), LB);
       ++r) {
    if (is_config_valid_inA(r, correct_power_of_two, LB,
                            LB * NUMBER_OF_STATES)) {
      std::vector<unsigned short int> this_conf(LB, 0);
      int_to_bosons(r, correct_power_of_two, LB, this_conf);
      int nright = 0;
      for (int p = 0; p < LB; ++p) {
        nright += numberparticle_for_basis_state[this_conf[p]];
      }
      int parity_right=nright%Q_;

      int parity_left = (ZN_Sector + Q_ - parity_right)%Q_;

      std::map<std::pair<int, int>, int>::const_iterator it =
          particle_sector.find(std::make_pair(parity_left, parity_right));
      if (it != particle_sector.end()) {
        if ((InverseMapB[it->second]).count(this_conf) == 0) {
          Confs_in_B[it->second].push_back(this_conf);
          InverseMapB[it->second][this_conf] = number_conf_in_B[it->second];
          number_conf_in_B[it->second] += 1;
        }
      }
    }
  }

  total_number_of_confs = 0;
  for (int nsa = 0; nsa < valid_sectors; ++nsa) {
    total_number_of_confs += Confs_in_A[nsa].size() * Confs_in_B[nsa].size();
  }

  starting_conf[0] = 0;
  for (int nsa = 1; nsa < valid_sectors; ++nsa) {
    starting_conf[nsa] =
        starting_conf[nsa - 1] +
        Confs_in_A[nsa - 1].size() * Confs_in_B[nsa - 1].size();
  }

  /** Now provide two indices to test inversion ***/

  std::stringstream ss;
  ss << "L=" << L << ".Q=" << Q_ << ".Zn_sector=" << ZN_Sector;
  string_names = ss.str();
}

void basis::change_basis(const basis& b0, const PetscScalar* state,
                         PetscScalar* new_state) {
  int LA0 = b0.LA;
  int LB0 = b0.LB;
  int conf_idx = 0;
  Conf cur_conf(L, 0);
  Conf cA(LA, 0);
  Conf cB(LB, 0);
  Conf cA0(LA0, 0);
  Conf cB0(LB0, 0);
  for (int nsa = 0; nsa < valid_sectors; ++nsa) {
    for (int p = 0; p < Confs_in_A[nsa].size(); ++p) {
      cA = Confs_in_A[nsa][p];
      for (int q = 0; q < Confs_in_B[nsa].size(); ++q) {
        cB = Confs_in_B[nsa][q];
        // std::cout << "# New B part: ";
        // for (int ib=0;ib<cB.size();ib++) std::cout << cB[ib];
        // std::cout << std::endl;
        // update cur_conf
        for (int i = 0; i < LA; i++) cur_conf[i] = cA[i];
        for (int j = 0; j < LB; j++) cur_conf[LA + j] = cB[j];
        // std::cout << "# Reading conf ";
        // for (int ib=0;ib<cur_conf.size();ib++) std::cout << cur_conf[ib];
        // std::cout << std::endl;
        // separate at the cut and count the number of particles in left/right
        // subsystems
        int nleft = 0, nright = 0;
        for (int i = 0; i < LA0; i++) {
          cA0[i] = cur_conf[i];
          nleft += cur_conf[i];
        }
        for (int j = 0; j < LB0; j++) {
          cB0[j] = cur_conf[LA0 + j];
          nright += cur_conf[LA0 + j];
        }
        // std::cout << "# A part: ";
        // for (int ib=0;ib<cA0.size();ib++) std::cout << cA0[ib];
        // std::cout << std::endl;
        // std::cout << "# B part: ";
        // for (int ib=0;ib<cB0.size();ib++) std::cout << cB0[ib];
        // std::cout << std::endl;
        // compute the sector index
        int nsa0 = b0.particle_sector.at(std::make_pair(nleft, nright));
        // compute the index (in b0) of the configuration
        int j1 = b0.InverseMapA[nsa0].at(cA0);
        int j2 = b0.InverseMapB[nsa0].at(cB0);
        int j = b0.starting_conf[nsa0] + j1 * b0.Confs_in_B[nsa0].size() + j2;
        // std::cout << "# For conf " << conf_idx << ", the old index was " << j
        // << std::endl; perform the permutation
        new_state[conf_idx] = state[j];
        conf_idx++;
      }
    }
  }
}

unsigned long int basis::index(Conf conf) {
  /*
   * Return the index (in the basis) of a given conf
   */
  Conf confA(conf.begin(), conf.begin() + LA);
  Conf confB(conf.begin() + LA, conf.end());
  // count the number of particles in the left and right subsystems
  int nleft = 0, nright = 0;
  for (auto& Sz : confA) nleft += Sz;
  for (auto& Sz : confB) nright += Sz;
  // sector number
  int nsa = particle_sector[std::make_pair(nleft%Q_, nright%Q_)];
  // index
  unsigned long int iconf = starting_conf[nsa] +
                            InverseMapA[nsa][confA] * Confs_in_B[nsa].size() +
                            InverseMapB[nsa][confB];
  return iconf;
}

void basis::get_conf_coefficients(unsigned long int index, int& nsa, int& ca,
                                  int& cb) {
  // using the fact that index = cb + LB*(ca + LA*nsa)
  // int reduced_index = index/LB;
  // cb = index%LB;
  // nsa = reduced_index/LA;
  // ca = reduced_index%LA;
  unsigned long int i = 0;
  // int NumB;
  // int NumA;
  for (nsa = 0; nsa < valid_sectors;
       ++nsa) {  // int nsb=basis_pointer->partner_sector[nsa];
    for (ca = 0; ca < Confs_in_A[nsa].size(); ++ca) {
      for (cb = 0; cb < Confs_in_B[nsa].size(); ++cb) {
        if (i == index) return;
        i++;
      }
    }
  }
}

void basis::init_vectors_product_state(Vec& Psi_t, Vec& Psi_t2, int i0) {
  // add 1 at position i0 in vector Psi_t
  VecSetValue(Psi_t, i0, 1.0, INSERT_VALUES);

  int i = 0;

  for (int nsa = 0; nsa < valid_sectors;
       ++nsa) {  // int nsb=partner_sector[nsa];
    for (int ca = 0; ca < Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < Confs_in_B[nsa].size(); ++cb) {
        if (i == i0) {
          // prefactor = state of the first spin of configuration B(nsa, cb)
          PetscScalar prefactor =1.0* Confs_in_B[nsa][cb][0];
          VecSetValue(Psi_t2, i0, prefactor, INSERT_VALUES);
        }
        i++;
      }
    }
  }
}

void basis::init_vectors_random(Vec& Psi_t, Vec& Psi_t2, int seed3,
                                PetscInt Istart, PetscInt Iend) {
  int i = 0;
  boost::mt19937 random_generator(seed3);
  boost::random::uniform_real_distribution<double> box_dist(0, 1);
  PetscScalar aa = 0;
  PetscScalar bb = 0;
  PetscScalar val;
  double mynorm = 0.;
  double mynorm2 = 0.;
  for (int nsa = 0; nsa < valid_sectors;
       ++nsa) {  // int nsb=partner_sector[nsa];
    for (int ca = 0; ca < Confs_in_A[nsa].size(); ++ca) {
      for (int cb = 0; cb < Confs_in_B[nsa].size(); ++cb) {
        while (aa == 0) {
          aa = box_dist(random_generator);
        }
        bb = box_dist(random_generator);
        val = sqrt(-2.0 * log(aa)) * cos(2 * PI * bb);
        while (aa == 0) {
          aa = box_dist(random_generator);
        }
        bb = box_dist(random_generator);
        //	val+=PETSC_i*sqrt(-2.0*log(aa))*cos(2*PI*bb);
        //	mynorm+=val*PetscConjComplex(val);
        if ((i >= Istart) && (i < Iend)) {
          VecSetValue(Psi_t, i, val, INSERT_VALUES);
          // measure n(L/2) over this state...
          PetscScalar prefactor =1.0*(Confs_in_B[nsa][cb][0]);
          VecSetValue(Psi_t2, i, val * prefactor, INSERT_VALUES);
          // std::cout << "VEC " << i << " " << val << " " << val*prefactor <<
          // endl; mynorm+=val*prefactor*
        }
        i++;
      }
    }
  }
}

void basis::init_vector_random(Vec& Psi_t, int seed3, PetscInt Istart,
                               PetscInt Iend) {
  /*
   * Vector with Iend - Istart + 1 random entries
   */
  boost::mt19937 random_generator(seed3);
  boost::random::uniform_real_distribution<double> box_dist(0, 1);
  PetscScalar aa = 0;
  PetscScalar bb = 0;
  PetscScalar val;

  for (int i = 0; i < total_number_of_confs; ++i) {
    while (aa == 0) {
      aa = box_dist(random_generator);
    }
    bb = box_dist(random_generator);
    val = sqrt(-2.0 * log(aa)) * cos(2 * PI * bb);
    while (aa == 0) {
      aa = box_dist(random_generator);
    }
    bb = box_dist(random_generator);
    if ((i >= Istart) && (i < Iend)) {
      VecSetValue(Psi_t, i, val, INSERT_VALUES);
    }
  }
}

void basis::int_to_bosons(unsigned long int num, int correct_power_of_two,
                          int L, std::vector<unsigned short int>& bosons) {
  boost::dynamic_bitset<> bb(correct_power_of_two * L, num);
  for (int r = 0; r < L; ++r) {
    bosons[r] = 0;
    for (int x = 0; x < correct_power_of_two; ++x) {
      bosons[r] += pow(2, x) * bb[correct_power_of_two * r + x];
    }
  }
}

bool basis::is_config_valid_inA(int num, int correct_power_of_two, int L,
                                int Nmax) {
  boost::dynamic_bitset<> bb(correct_power_of_two * L, num);
  int total_n = 0;

  for (int r = 0; r < L; ++r) {
    int nn = 0;
    for (int x = 0; x < correct_power_of_two; ++x) {
      nn += pow(2, x) * bb[correct_power_of_two * r + x];
    }
    if (nn > (NUMBER_OF_STATES - 1)) {
      return 0;
    }
    total_n += nn;
  }
  if (total_n > Nmax) {
    return 0;
  }
  return 1;
}

#endif
