#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

class Hamiltonian
{

 private:

  basis * basis_pointer;
  Mat * pointer_to_H;
 public:

  Hamiltonian(basis * this_basis_pointer, Mat * matrix_pointer);
  ~Hamiltonian() {}


  void get_parameters();


    void init_matrix_sizes(PetscInt Istart, PetscInt Iend);
    void create_matrix(PetscInt Istart, PetscInt Iend);

    void create_matrix_lapack(double *A);
    void diagonalize_lapack(double *A, double w[], bool eigenvectors);
    std::string string_names;

    int L; int LA; int LB;
    double Gamma;double disorder;  double J; double field_disorder;
     std::vector<double> coupling;  PetscBool pbc;
     std::vector<double> field;
     double J2;
    PetscInt  seed;
    double g; double g2;
    double V_;


    int myrank;
    short int NUMBER_OF_STATES;
      short int Q_;
      std::vector<PetscComplex> diag_phase_factor;
      std::vector<PetscComplex> diag_phase_factor_cc;
  };




Hamiltonian::Hamiltonian(basis * this_basis_pointer, Mat * matrix_pointer)
{
    basis_pointer=this_basis_pointer;
    pointer_to_H=matrix_pointer;
    L=basis_pointer->L;
    LA=basis_pointer->LA;
    LB=basis_pointer->LB;
    myrank = basis_pointer->myrank;
    NUMBER_OF_STATES = basis_pointer->NUMBER_OF_STATES;
//    Q_=basis_pointer->Q_;
  Q_=basis_pointer->Q_;

  diag_phase_factor.resize(Q_);
  diag_phase_factor_cc.resize(Q_);

  diag_phase_factor[0]=1.; diag_phase_factor_cc[0]=1.;
  for (int q=1;q<Q_;++q) {
    diag_phase_factor[q]=diag_phase_factor[q-1]*exp(2*PI*PETSC_i/Q_);
    diag_phase_factor_cc[q]=diag_phase_factor_cc[q-1]*exp(-2*PI*PETSC_i/Q_);
}

}



void Hamiltonian::get_parameters() {

    J=1.0; seed=0; Gamma=0.;
    disorder=0.; field_disorder=0.;
     pbc=PETSC_TRUE;
     seed=0;
     J2=0.; g=0; g2=0.; V_=0;


  PetscErrorCode ierr;
  PetscOptionsGetReal(NULL,NULL,"-J",&J,NULL);
  PetscOptionsGetReal(NULL,NULL,"-J2",&J2,NULL);
  PetscOptionsGetReal(NULL,NULL,"-g",&g,NULL);
  PetscOptionsGetReal(NULL,NULL,"-g2",&g2,NULL);
  PetscOptionsGetReal(NULL,NULL,"-V_",&V_,NULL);
  PetscOptionsGetReal(NULL,NULL,"-Gamma",&Gamma,NULL);
  PetscOptionsGetReal(NULL,NULL,"-disorder",&disorder,NULL);
  PetscOptionsGetReal(NULL,NULL,"-field_disorder",&field_disorder,NULL);

  PetscOptionsGetBool(NULL,NULL,"-pbc",&pbc,NULL);
  PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL);

  coupling.resize(L,J);

  if (disorder>0)
    {
      boost::mt19937 generator; generator.seed(seed);
      boost::random::uniform_real_distribution<double> box(-disorder,disorder);

	for(int i=0; i<L; i++) { coupling[i]+= box(generator); }

    }
    if (!(pbc)) { coupling[L-1]=0.;}
    field.resize(L,Gamma);
    if (field_disorder>0)
    {
      boost::mt19937 generator; generator.seed(seed+50);
      boost::random::uniform_real_distribution<double> box(-field_disorder,field_disorder);

    for(int i=0; i<L; i++) { field[i] = box(generator); }

    }


    if (myrank==0) {
    std::cout << "# coupling= { ";  for(int i=0; i<L; i++) { std::cout << coupling[i] << " ";} std::cout <<" }" << endl;
    std::cout << "# field= { ";  for(int i=0; i<L; i++) { std::cout << field[i] << " ";} std::cout <<" }" << endl;
    std::cout << "# J2= " << J2 << endl;
    std::cout << "# g= " << g << endl;
    std::cout << "# g2= " << g2 << endl;
    }


  std::stringstream ss;
  ss << ".Gamma=" << Gamma;
  if (J!=1) { ss << ".J=" << J;}
  if (J2!=0) { ss << ".J2=" << J2;}
  if (g!=0) { ss << ".g=" << g;}
  if (g2!=0) { ss << ".g2=" << g2;}
  if (V_!=0) { ss << ".V=" << V_;}
  if (disorder>0) { ss << ".disorder=" << disorder;ss << ".seed=" << seed; }
  string_names=ss.str();

}



void Hamiltonian::init_matrix_sizes(PetscInt Istart, PetscInt Iend) {

std::vector<PetscInt> d_nnz; std::vector<PetscInt> o_nnz;
size_t this_size=Iend-Istart; d_nnz.resize(this_size,0); o_nnz.resize(this_size,0);


  int row_ctr=0;
for (int nsa=0;nsa<basis_pointer->valid_sectors;++nsa) {
  //   int nsb=basis_pointer->partner_sector[nsa];
    for (int ca=0;ca<basis_pointer->Confs_in_A[nsa].size();++ca) {
	std::vector<unsigned short int> config(L,0);
	std::vector<unsigned short int> confA=basis_pointer->Confs_in_A[nsa][ca];
	for (int r=0;r<LA;++r) { config[r]=confA[r]; }
	    for (int cb=0;cb<basis_pointer->Confs_in_B[nsa].size();++cb) {
	      //    cout << " *** cb=" << cb << endl;
	      if ( (row_ctr>=Istart) && (row_ctr<Iend) ) {
		// construct B-part only if in the good range ...
	      std::vector<unsigned short int> confB=basis_pointer->Confs_in_B[nsa][cb];
	      for (int r=0;r<LB;++r) { config[r+LA]=confB[r]; }
	      std::vector<unsigned short int> newconfig=config;
	      std::vector<unsigned short int> newconfA=confA;
	      std::vector<unsigned short int> newconfB=confB;
	      int j;
	      double diag=0.;
	      for (int r=0;r<L;++r) {

		int rb=r; int rb2=(r+1)%L; int rb3=(r+2)%L;
    if (field[r]!=0) {
      diag+= (-field[r]) * 2*cos(2.0*PI/Q_*config[rb]);
    }
    if (g2!=0) {
      diag += -g2*4*sin(2.0*PI/Q_*config[rb]);
    }
    /*
    if (V_!=0) {
      diag+= (-V_) * 2*cos(2.0*PI/Q_*config[rb]);
    }
*/
    if (coupling[r] || (g!=0)) {
    for (int plus=-1;plus<=1;plus+=2) {
    newconfig[rb]=(config[rb]+plus+Q_)%Q_;
    newconfig[rb2]=(config[rb2]-plus+Q_)%Q_;
		  int nleft=0; int nright=0;
		for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
		int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft%Q_,nright%Q_)];

		j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		if ((j>=Iend) or (j<Istart)) { o_nnz[row_ctr-Istart]++;} else { d_nnz[row_ctr-Istart]++;}

    newconfig[rb]=config[rb];
    newconfig[rb2]=config[rb2];
  } }
    // J2
    if (J2!=0) {
    for (int plus=-1;plus<=1;plus+=2) {
    newconfig[rb]=(config[rb]+plus+Q_)%Q_;
    newconfig[rb3]=(config[rb3]-plus+Q_)%Q_;
		  int nleft=0; int nright=0;
		for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
		int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft%Q_,nright%Q_)];

		j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		if ((j>=Iend) or (j<Istart)) { o_nnz[row_ctr-Istart]++;} else { d_nnz[row_ctr-Istart]++;}
    }
    newconfig[rb]=config[rb];
    newconfig[rb3]=config[rb3];
		}




  }
		// assume always a diagonal part ...
	      if (diag!=0) {
	      d_nnz[row_ctr-Istart]++;
	      }
	      }

	      row_ctr++;
	    } // loop over cb
	  } // over ca
 } // over nsA sectors


        MatCreateAIJ( PETSC_COMM_WORLD, Iend-Istart, PETSC_DECIDE, basis_pointer->total_number_of_confs, basis_pointer->total_number_of_confs, 0, d_nnz.data(), 0, o_nnz.data(), &(*pointer_to_H));


}


void Hamiltonian::create_matrix(PetscInt Istart, PetscInt Iend) {


        int row_ctr=0; int j=0;
	for (int nsa=0;nsa<basis_pointer->valid_sectors;++nsa) {
	  for (int ca=0;ca<basis_pointer->Confs_in_A[nsa].size();++ca) {
	    std::vector<unsigned short int> config(L,0);
	    std::vector<unsigned short int> confA=basis_pointer->Confs_in_A[nsa][ca];
	    for (int r=0;r<LA;++r) { config[r]=confA[r];}
	    for (int cb=0;cb<basis_pointer->Confs_in_B[nsa].size();++cb) {
	      if ( (row_ctr>=Istart) && (row_ctr<Iend) ) {
		// construct B-part only if in the good range ...
	      std::vector<unsigned short int> confB=basis_pointer->Confs_in_B[nsa][cb];
              for (int r=0;r<LB;++r) { config[r+LA]=confB[r];  }
	      // now do Hamiltonian basis size ...
	      std::vector<unsigned short int> newconfig=config;
	      std::vector<unsigned short int> newconfA=confA;
	      std::vector<unsigned short int> newconfB=confB;
	      int j;
	      double diag=0.;
  	      for (int r=0;r<L;++r) {

  		int rb=r; int rb2=(r+1)%L; int rb3=(r+2)%L;

      if (field[r]!=0) {
        diag+= -field[r] * 2*cos(2.0*PI/Q_*config[rb]);
      }
      if (g2!=0) {
        diag += -g2*4*sin(2.0*PI/Q_*config[rb]);
      }

      if (coupling[r] || (g!=0)) {
      for (int plus=-1;plus<=1;plus+=2) {
      newconfig[rb]=(config[rb]+plus+Q_)%Q_;
      newconfig[rb2]=(config[rb2]-plus+Q_)%Q_;

		  int nleft=0; int nright=0;
		for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
		int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft%Q_,nright%Q_)];
		j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		MatSetValue(*pointer_to_H, row_ctr, j, (PetscScalar) -coupling[r], ADD_VALUES );
    if (plus==1) { // sigma. sigma^dag
      MatSetValue(*pointer_to_H, row_ctr, j, (PetscScalar) -2*g*sin(2.0*PI/Q_*newconfig[rb]), ADD_VALUES );

		}
    if (plus==-1) {
      MatSetValue(*pointer_to_H, row_ctr, j, (PetscScalar) -2*g*sin(2.0*PI/Q_*config[rb]), ADD_VALUES );

    }

        newconfig[rb]=config[rb];
        newconfig[rb2]=config[rb2];

  } }
  // J2
  if (J2!=0) {
  for (int plus=-1;plus<=1;plus+=2) {
  newconfig[rb]=(config[rb]+plus+Q_)%Q_;
  newconfig[rb3]=(config[rb3]-plus+Q_)%Q_;
    int nleft=0; int nright=0;
  for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
  for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
  int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft%Q_,nright%Q_)];

  j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
  MatSetValue(*pointer_to_H, row_ctr, j, (PetscScalar) -J2, ADD_VALUES );
  }
  newconfig[rb]=config[rb];
  newconfig[rb3]=config[rb3];
  }



}
		// assume always a diagonal part ...
	      if (diag!=0) {
		MatSetValue(*pointer_to_H, row_ctr, row_ctr, (PetscScalar) diag, ADD_VALUES ); //d_nnz[row_ctr-Istart]++;
	      }

	      }
	      row_ctr++;
	    } // loop over cb
	  } // over ca
	} // over nsA sectors
}


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
	      double diag=0.;
  	      for (int r=0;r<L;++r) {

  		int rb=r; int rb2=(r+1)%L; int rb3=(r+2)%L;

      if (field[r]!=0) {
        diag+= -field[r] * 2*cos(2.0*PI/Q_*config[rb]);
      }

      if (coupling[r] || (g!=0)) {
      for (int plus=-1;plus<=1;plus+=2) {
      newconfig[rb]=(config[rb]+plus+Q_)%Q_;
      newconfig[rb2]=(config[rb2]-plus+Q_)%Q_;

		  int nleft=0; int nright=0;
		for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
		int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft%Q_,nright%Q_)];
		j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		A[j * nconf + row_ctr] +=  (-coupling[r]);
    if (plus==1) { // sigma. sigma^dag
      A[j * nconf + row_ctr] += (-2*g*sin(2.0*PI/Q_*newconfig[rb]));

		}
    if (plus==-1) {
      A[j * nconf + row_ctr] += (-2*g*sin(2.0*PI/Q_*config[rb]));
    }

        newconfig[rb]=config[rb];
        newconfig[rb2]=config[rb2];

  } }
  // J2
  if (J2!=0) {
  for (int plus=-1;plus<=1;plus+=2) {
  newconfig[rb]=(config[rb]+plus+Q_)%Q_;
  newconfig[rb3]=(config[rb3]-plus+Q_)%Q_;
    int nleft=0; int nright=0;
  for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
  for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
  int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft%Q_,nright%Q_)];

  j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
  A[j * nconf + row_ctr] += (-J2);
  }
  newconfig[rb]=config[rb];
  newconfig[rb3]=config[rb3];
  }



}
		// assume always a diagonal part ...
	      if (diag!=0) {
          A[row_ctr * nconf + row_ctr] += diag; //d_nnz[row_ctr-Istart]++;
	      }
        row_ctr++;

      }  // loop over cb
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
  // tell the routine whether or not to compute the eigenvectors
  char *vectors = "N";
  if (eigenvectors) vectors = "V";
  dsyevd(vectors, "Lower", &myn, &A[0], &lda, w, &wkopt, &lwork, &iwkopt,
         &liwork, &info);
  // Allocate the optimal workspace
  lwork = (MKL_INT)wkopt;
  work = (double *)malloc(lwork * sizeof(double));
  liwork = iwkopt;
  iwork = (MKL_INT *)malloc(liwork * sizeof(MKL_INT));
  // Perform the diagonalization
  dsyevd(vectors, "Lower", &myn, &A[0], &lda, w, work, &lwork, iwork, &liwork,
         &info);
  free((void *)iwork);
  free((void *)work);
  // return w;
}

#endif
