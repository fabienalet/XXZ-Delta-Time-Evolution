#ifndef PARAM_H
#define PARAM_H

using namespace std;

class Parameters {
  /*
    Read the parameters and open files
  */
 private:
 public:
  Parameters(int myrank_);
  ~Parameters() {}

  void Initialize_timegrid();
  void init_filenames(ofstream& entout, ofstream& imbout, ofstream& locout,
                      ofstream& retout, ofstream& partout);
  void init_filenames(ofstream& entout, ofstream& imbout, ofstream& locout,
                      ofstream& retout, ofstream& partout, int init_conf);
  void init_filenames(ofstream& entout, ofstream& entcutout, ofstream& imbout,
                      ofstream& locout, ofstream& retout, ofstream& partout,
                      int init_conf);
  void init_filenames_eigenstate_full(ofstream& entout, ofstream& locout,
                                      ofstream& partout);
  void init_filenames_eigenstate(ofstream& entout, ofstream& locout,
                                 ofstream& partout, double target);
  void init_filenames_eigenstate(ofstream& entout, ofstream& locout,
                                 ofstream& partout, ofstream& parthistogramout,
                                 double target);
  void init_filenames_eigenstate(ofstream& entout, ofstream& locout,
                                 ofstream& partout, ofstream& corrout,
                                 ofstream& tcorrout,
                                 double target);
  void init_filenames_energy(ofstream& enout, ofstream& rgapout, double target);
  std::vector<double> time_points;
  std::vector<double> delta_t_points;

  int myrank;
  PetscInt L;
  PetscInt LA;
  PetscInt LB;
  PetscInt num_times;
  PetscReal Tmin;
  PetscReal Tmax;
  PetscReal tcoupling;
  PetscBool use_linear_timegrid;
  PetscBool loggrid;
  PetscReal dt;
  // PetscInt i01;   PetscInt i02;

  PetscBool cdw_start;
  PetscBool product_state_start;
  PetscInt num_product_states;
  PetscReal TEEmin;
  PetscReal TEEmax;
  PetscInt nmeasures;

  PetscBool measure_entanglement;
  PetscBool measure_local;
  PetscBool measure_entanglement_at_all_cuts;
  PetscBool measure_imbalance;
  PetscBool measure_returnE;
  PetscBool measure_participation;
  PetscBool measure_correlations;
  PetscBool measure_transverse_correlations;
  PetscBool measure_all_part_entropy;
  double qmin;
  double qmax;
  int Nq;

  PetscBool measure_eigenvector_largest_coefficients;
  int Ncoeffs_elc;
  PetscBool measure_eigenvector_largest_coefficients_with_sign;

  PetscBool measure_return;
  PetscBool eigenvectors;
  PetscBool target_infinite_temperature;
  PetscBool write_wf;
  PetscBool targets_set;
  std::vector<double> targets;

  std::string string_from_H;
  std::string string_from_basis;
};

void Parameters::init_filenames(ofstream& entout, ofstream& imbout,
                                ofstream& locout, ofstream& retout,
                                ofstream& partout) {
  if (measure_entanglement) {
    std::stringstream filename;
    filename << "ENT." << string_from_basis << string_from_H << ".dat";
    entout.open((filename.str()).c_str());
    entout.precision(20);
  }

  if (measure_imbalance) {
    std::stringstream imbfilename;
    imbfilename << "Imbalance." << string_from_basis << string_from_H << ".dat";
    imbout.open((imbfilename.str()).c_str());
    imbout.precision(20);
  }

  if (measure_local) {
    std::stringstream locfilename;
    locfilename << "LocalObservable." << string_from_basis << string_from_H
                << ".dat";
    locout.open((locfilename.str()).c_str());
    locout.precision(20);
  }

  if (measure_return) {
    std::stringstream retfilename;
    retfilename << "Return." << string_from_basis << string_from_H << ".dat";
    retout.open((retfilename.str()).c_str());
    retout.precision(20);
  }

  if (measure_participation) {
    std::stringstream partfilename;
    partfilename << "Participation." << string_from_basis << string_from_H
                 << ".dat";
    partout.open((partfilename.str()).c_str());
    partout.precision(20);
  }
}

void Parameters::init_filenames(ofstream& entout, ofstream& imbout,
                                ofstream& locout, ofstream& retout,
                                ofstream& partout, int init_conf) {
  if (measure_entanglement) {
    std::stringstream filename;
    filename << "ENT." << string_from_basis << string_from_H
             << ".init_conf=" << init_conf << ".dat";
    entout.open((filename.str()).c_str());
    entout.precision(20);
  }

  if (measure_imbalance) {
    std::stringstream imbfilename;
    imbfilename << "Imbalance." << string_from_basis << string_from_H
                << ".init_conf=" << init_conf << ".dat";
    imbout.open((imbfilename.str()).c_str());
    imbout.precision(20);
  }

  if (measure_local) {
    std::stringstream locfilename;
    locfilename << "LocalObservable." << string_from_basis << string_from_H
                << ".init_conf=" << init_conf << ".dat";
    locout.open((locfilename.str()).c_str());
    locout.precision(20);
  }

  if (measure_return) {
    std::stringstream retfilename;
    retfilename << "Return." << string_from_basis << string_from_H
                << ".init_conf=" << init_conf << ".dat";
    retout.open((retfilename.str()).c_str());
    retout.precision(20);
  }

  if (measure_participation) {
    std::stringstream partfilename;
    partfilename << "Participation." << string_from_basis << string_from_H
                 << ".init_conf=" << init_conf << ".dat";
    partout.open((partfilename.str()).c_str());
    partout.precision(20);
  }
}

void Parameters::init_filenames(ofstream& entout, ofstream& entcutout,
                                ofstream& imbout, ofstream& locout,
                                ofstream& retout, ofstream& partout,
                                int init_conf) {
  if (measure_entanglement) {
    std::stringstream filename;
    filename << "ENT." << string_from_basis << string_from_H
             << ".init_conf=" << init_conf << ".dat";
    entout.open((filename.str()).c_str());
    entout.precision(20);
  }

  if (measure_entanglement_at_all_cuts) {
    std::stringstream filename;
    filename << "ENT.cuts." << string_from_basis << string_from_H
             << ".init_conf=" << init_conf << ".dat";
    entcutout.open((filename.str()).c_str());
    entcutout.precision(20);
  }

  if (measure_imbalance) {
    std::stringstream imbfilename;
    imbfilename << "Imbalance." << string_from_basis << string_from_H
                << ".init_conf=" << init_conf << ".dat";
    imbout.open((imbfilename.str()).c_str());
    imbout.precision(20);
  }

  if (measure_local) {
    std::stringstream locfilename;
    locfilename << "LocalObservable." << string_from_basis << string_from_H
                << ".init_conf=" << init_conf << ".dat";
    locout.open((locfilename.str()).c_str());
    locout.precision(20);
  }

  if (measure_return) {
    std::stringstream retfilename;
    retfilename << "Return." << string_from_basis << string_from_H
                << ".init_conf=" << init_conf << ".dat";
    retout.open((retfilename.str()).c_str());
    retout.precision(20);
  }

  if (measure_participation) {
    std::stringstream partfilename;
    partfilename << "Participation." << string_from_basis << string_from_H
                 << ".init_conf=" << init_conf << ".dat";
    partout.open((partfilename.str()).c_str());
    partout.precision(20);
  }
}

void Parameters::init_filenames_eigenstate_full(ofstream& entout,
                                                ofstream& locout,
                                                ofstream& partout) {
  if (measure_entanglement) {
    std::stringstream filename;
    filename << "ENT." << string_from_basis << string_from_H << ".dat";
    entout.open((filename.str()).c_str());
    entout.precision(20);
  }

  if (measure_local) {
    std::stringstream locfilename;
    locfilename << "LocalObservable." << string_from_basis << string_from_H
                << ".dat";
    locout.open((locfilename.str()).c_str());
    locout.precision(20);
  }

  if (measure_participation) {
    std::stringstream partfilename;
    partfilename << "Participation." << string_from_basis << string_from_H
                 << ".dat";
    partout.open((partfilename.str()).c_str());
    partout.precision(20);
  }
}

void Parameters::init_filenames_eigenstate(ofstream& entout, ofstream& locout,
                                           ofstream& partout, double target) {
  if (measure_entanglement) {
    std::stringstream filename;
    filename << "ENT." << string_from_basis << string_from_H << ".target"
             << target << ".dat";
    entout.open((filename.str()).c_str());
    entout.precision(20);
  }

  if (measure_local) {
    std::stringstream locfilename;
    locfilename << "LocalObservable." << string_from_basis << string_from_H
                << ".target" << target << ".dat";
    locout.open((locfilename.str()).c_str());
    locout.precision(20);
  }

  if (measure_participation) {
    std::stringstream partfilename;
    partfilename << "Participation." << string_from_basis << string_from_H
                 << ".target" << target << ".dat";
    partout.open((partfilename.str()).c_str());
    partout.precision(20);
  }
}

void Parameters::init_filenames_eigenstate(ofstream& entout, ofstream& locout,
                                           ofstream& partout,
                                           ofstream& parthistout,
                                           double target) {
  if (measure_entanglement) {
    std::stringstream filename;
    filename << "ENT." << string_from_basis << string_from_H << ".target"
             << target << ".dat";
    entout.open((filename.str()).c_str());
    entout.precision(20);
  }

  if (measure_local) {
    std::stringstream locfilename;
    locfilename << "LocalObservable." << string_from_basis << string_from_H
                << ".target" << target << ".dat";
    locout.open((locfilename.str()).c_str());
    locout.precision(20);
  }

  if (measure_participation) {
    std::stringstream partfilename;
    partfilename << "Participation." << string_from_basis << string_from_H
                 << ".target" << target << ".dat";
    partout.open((partfilename.str()).c_str());
    partout.precision(20);
  }

  if (measure_eigenvector_largest_coefficients) {
    std::stringstream parthistfilename;
    parthistfilename << "EigenVectorLargestCoefficients." << string_from_basis
                     << string_from_H << ".dat";
    parthistout.open((parthistfilename.str()).c_str());
    parthistout.precision(20);
  }

  if (measure_eigenvector_largest_coefficients_with_sign) {
    std::stringstream parthistfilename;
    parthistfilename << "EigenVectorLargestCoefficientsWithSign."
                     << string_from_basis << string_from_H << ".dat";
    parthistout.open((parthistfilename.str()).c_str());
    parthistout.precision(20);
  }

  if (measure_correlations) {
    std::stringstream parthistfilename;
    parthistfilename << "Correl."
                     << string_from_basis << string_from_H << ".dat";
    parthistout.open((parthistfilename.str()).c_str());
    parthistout.precision(20);
  }

}


void Parameters::init_filenames_eigenstate(ofstream& entout, ofstream& locout,
                                           ofstream& partout,
                                           ofstream& corrout, 
                                           ofstream& tcorrout,
                                           double target) {
  if (measure_entanglement) {
    std::stringstream filename;
    filename << "ENT." << string_from_basis << string_from_H << ".target"
             << target << ".dat";
    entout.open((filename.str()).c_str());
    entout.precision(20);
  }


  if (measure_local) {
    std::stringstream locfilename;
    locfilename << "LocalObservable." << string_from_basis << string_from_H
                << ".target" << target << ".dat";
    locout.open((locfilename.str()).c_str());
    locout.precision(20);
  }

  if (measure_participation) {
    std::stringstream partfilename;
    partfilename << "Participation." << string_from_basis << string_from_H
                 << ".target" << target << ".dat";
    partout.open((partfilename.str()).c_str());
    partout.precision(20);
  }

  if (measure_correlations) {
    std::stringstream corrfilename;
    corrfilename << "Correl."
                     << string_from_basis << string_from_H << ".dat";
    corrout.open((corrfilename.str()).c_str());
    corrout.precision(20);
  }

  if (measure_transverse_correlations) {
    std::stringstream tcorrfilename;
    tcorrfilename << "TransverseCorrel."
                  << string_from_basis << string_from_H << ".dat";
    tcorrout.open((tcorrfilename.str()).c_str());
    tcorrout.precision(20);
  }

}

void Parameters::init_filenames_energy(ofstream& enout, ofstream& rgapout,
                                       double target) {
  std::stringstream filename;
  filename << "Energies." << string_from_basis << string_from_H << ".target"
           << target << ".dat";
  enout.open((filename.str()).c_str());
  enout.precision(20);

  std::stringstream gapfilename;
  gapfilename << "Rgap." << string_from_basis << string_from_H << ".target"
              << target << ".dat";
  rgapout.open((gapfilename.str()).c_str());
  rgapout.precision(20);
}

void Parameters::Initialize_timegrid() {
  double b = 1;
  for (int kk = 0; kk <= num_times; ++kk) {
    double Tj, Deltat;
    if (false == use_linear_timegrid) {
      b = exp(log(Tmax / Tmin) / static_cast<double>(num_times));
      Tj = Tmin * pow(b, kk);
      Deltat;
      if (kk == 0) {
        Deltat = Tj;
      } else {
        Deltat = Tj * (1. - 1. / b);
      }
    } else {
      if (kk == 0) {
        Deltat = Tmin;
        Tj = Tmin;
      } else {
        Deltat = (Tmax - Tmin) / static_cast<double>(num_times);
        Tj = Tmin + kk * Deltat;
      }
    }
    time_points.push_back(Tj);
    delta_t_points.push_back(Deltat);
  }

  if (myrank == 0) {
    std::cout << "#Time evolution from " << Tmin << " to " << Tmax << ", with "
              << num_times << " points on a ";
    if (use_linear_timegrid) {
      std::cout << "linear";
    } else {
      std::cout << "logarithmic";
    }
    std::cout << " time grid\n";
  }
}

Parameters::Parameters(int myrank_) {
  myrank = myrank_;
  PetscErrorCode ierr;

  L = 6;
  ierr = PetscOptionsGetInt(NULL, NULL, "-L", &L, NULL);  // CHKERRQ(ierr);

  LA = L / 2;
  LB = L / 2;
  ierr = PetscOptionsGetInt(NULL, NULL, "-LA", &LA, NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-LB", &LB, NULL);  // CHKERRQ(ierr);

  num_times = 100;
  Tmin = 0.;
  Tmax = 1.;
  use_linear_timegrid = PETSC_TRUE;
  loggrid = PETSC_FALSE;
  // i01=-1; i02=-1;

  cdw_start = PETSC_FALSE;
  product_state_start = PETSC_FALSE;
  TEEmin = Tmin;
  TEEmax = Tmax;
  nmeasures = num_times;

  measure_entanglement = PETSC_FALSE;
  measure_entanglement_at_all_cuts = PETSC_FALSE;
  measure_imbalance = PETSC_FALSE;
  measure_return = PETSC_FALSE;
  measure_participation = PETSC_TRUE;
  measure_eigenvector_largest_coefficients = PETSC_FALSE;
  measure_eigenvector_largest_coefficients_with_sign = PETSC_FALSE;
  measure_all_part_entropy = PETSC_FALSE;
  measure_correlations = PETSC_FALSE;
  measure_transverse_correlations = PETSC_FALSE;
  measure_local = PETSC_FALSE;
  dt = 1.0;
  qmin = 1.;
  qmax = 2.;
  Nq = 1;
  Ncoeffs_elc = 128;

  ierr = PetscOptionsGetBool(NULL, NULL, "-measure_correlations",
                             &measure_correlations, NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-measure_transverse_correlations",
                              &measure_transverse_correlations, 
                               NULL);  // CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(NULL, NULL, "-num_times", &num_times,
                            NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-loggrid", &loggrid,
                             NULL);  // CHKERRQ(ierr);
  ierr =
      PetscOptionsGetReal(NULL, NULL, "-Tmax", &Tmax, NULL);  // CHKERRQ(ierr);
  ierr =
      PetscOptionsGetReal(NULL, NULL, "-Tmin", &Tmin, NULL);  // CHKERRQ(ierr);
  if (loggrid) {
    use_linear_timegrid = PETSC_FALSE;
    if (Tmin == 0) {
      Tmin = 1.;
    }
  }
  tcoupling = 0.;
  ierr = PetscOptionsGetReal(NULL, NULL, "-tcoupling", &tcoupling,
                             NULL);  // CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL, NULL, "-product_state_start",
                             &product_state_start, NULL);  // CHKERRQ(ierr);
  num_product_states = 1;
  ierr = PetscOptionsGetInt(NULL, NULL, "-num_product_states",
                            &num_product_states, NULL);  // CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-cdw_start", &cdw_start,
                             NULL);  // CHKERRQ(ierr);
  TEEmin = Tmin;
  TEEmax = Tmax;
  nmeasures = num_times;
  ierr = PetscOptionsGetInt(NULL, NULL, "-num_measures", &nmeasures, NULL);
  ierr = PetscOptionsGetBool(NULL, NULL, "-measure_entanglement",
                             &measure_entanglement, NULL);
  ierr = PetscOptionsGetBool(NULL, NULL, "-measure_entanglement_at_all_cuts",
                             &measure_entanglement_at_all_cuts, NULL);
  ierr = PetscOptionsGetBool(NULL, NULL, "-measure_imbalance",
                             &measure_imbalance, NULL);
  ierr =
      PetscOptionsGetBool(NULL, NULL, "-measure_return", &measure_return, NULL);
  ierr =
      PetscOptionsGetBool(NULL, NULL, "-measure_local", &measure_local, NULL);
  ierr = PetscOptionsGetBool(NULL, NULL, "-measure_all_part_entropy",
                             &measure_all_part_entropy, NULL);

  ierr = PetscOptionsGetBool(NULL, NULL,
                             "-measure_eigenvector_largest_coefficients",
                             &measure_eigenvector_largest_coefficients, NULL);
  ierr = PetscOptionsGetBool(
      NULL, NULL, "-measure_eigenvector_largest_coefficients_with_sign",
      &measure_eigenvector_largest_coefficients_with_sign, NULL);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Ncoeffs", &Ncoeffs_elc, NULL);

  eigenvectors = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-eigenvectors", &eigenvectors, NULL);
  target_infinite_temperature = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-target_infinite_temperature",
                             &target_infinite_temperature, NULL);

  ierr = PetscOptionsGetReal(NULL, NULL, "-qmin", &qmin, NULL);
  ierr = PetscOptionsGetReal(NULL, NULL, "-qmax", &qmax, NULL);
  ierr = PetscOptionsGetInt(NULL, NULL, "-Nq", &Nq, NULL);

  write_wf = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-write_wf", &write_wf, NULL);

  int each_measurement = num_times / nmeasures;

  char* targets_c_string = new char[1000];
  ierr = PetscOptionsGetString(NULL, NULL, "-targets", targets_c_string, 1000,
                               &targets_set);  // CHKERRQ(ierr);
  if (targets_set) {
    std::string targets_string(targets_c_string);
    std::stringstream tsstr;
    tsstr.str(targets_string);
    double tgt;
    while (tsstr >> tgt) {
      targets.push_back(tgt);
    }
    delete[] targets_c_string;
  } else {
    targets.push_back(0.5);
  }  // default target = 0.5
}

#endif
