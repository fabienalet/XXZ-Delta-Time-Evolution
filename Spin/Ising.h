#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

class Hamiltonian
{

 private:

  fullbasis * basis_pointer;
  Mat * pointer_to_H;
 public:

  Hamiltonian(fullbasis * this_basis_pointer, Mat * matrix_pointer);
  ~Hamiltonian() {}


  void get_parameters();


    void init_matrix_sizes(PetscInt Istart, PetscInt Iend);
    void create_matrix(PetscInt Istart, PetscInt Iend);
    std::string string_names;

    int L; int LA; int LB;
    double Gamma;
    double disorder;  
    double J; 
    double field_disorder;
    std::vector<double> coupling;  
    PetscBool pbc;
    std::vector<double> field;
    double g;
    PetscInt  seed;

    int myrank;
    short int NUMBER_OF_STATES;
    short int Q_;

    std::vector<double> diag_phase_factor;
    };




Hamiltonian::Hamiltonian(fullbasis * this_basis_pointer, Mat * matrix_pointer)
{
    basis_pointer=this_basis_pointer;
    pointer_to_H=matrix_pointer;
    L=basis_pointer->L;
    LA=basis_pointer->LA;
    LB=basis_pointer->LB;
    myrank = basis_pointer->myrank;
    NUMBER_OF_STATES = basis_pointer->NUMBER_OF_STATES;
    Q_=basis_pointer->NUMBER_OF_STATES;

  diag_phase_factor.resize(Q_);

  diag_phase_factor[0]=1.; diag_phase_factor[1]=-1.;

}



void Hamiltonian::get_parameters() {

    J=1.0; seed=0; Gamma=0.;
    disorder=0.; 
    pbc=PETSC_FALSE;
    seed=0; 
    g=0.;

  PetscOptionsGetReal(NULL,NULL,"-J",&J,NULL);
  PetscOptionsGetReal(NULL,NULL,"-Gamma",&Gamma,NULL);
  PetscOptionsGetReal(NULL,NULL,"-g",&g,NULL);
  PetscOptionsGetReal(NULL,NULL,"-disorder",&disorder,NULL);
  PetscOptionsGetReal(NULL,NULL,"-W",&disorder,NULL);
  PetscOptionsGetBool(NULL,NULL,"-pbc",&pbc,NULL);
  PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL);

  coupling.resize(L,J);
  field.resize(L,Gamma);

  if (disorder>0)
    {
      boost::mt19937 generator; generator.seed(seed);
      boost::random::uniform_real_distribution<double> box(0,disorder);
	    for(int i=0; i<L; i++) { coupling[i] += box(generator); }
      boost::mt19937 generator2; generator2.seed(seed+50);
      boost::random::uniform_real_distribution<double> box2(0,1./disorder);
      for(int i=0; i<L; i++) { field[i] = box2(generator2); }
    }
    if (!(pbc)) { coupling[L-1]=0.;}



  if (myrank==0) {
  std::cout << "# coupling= { ";  for(int i=0; i<L; i++) { std::cout << coupling[i] << " ";} std::cout <<" }" << endl;
  std::cout << "# field= { ";  for(int i=0; i<L; i++) { std::cout << field[i] << " ";} std::cout <<" }" << endl;
  std::cout << "# g= " << g << endl;
  }


  std::stringstream ss;
  if (Gamma!=0) { ss << ".Gamma=" << Gamma; }
  if (J!=1) { ss << ".J=" << J;}
  if (g!=0) { ss << ".g=" << g;}
  if (disorder>0) { ss << ".W=" << disorder;ss << ".seed=" << seed; }
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
	      PetscScalar diag=0.;

        // Let's work in the basis (call it x basis) where the main Ising interaction is diagonal
        // In this basis the transverse field is off-diagonal
         // J_i sx_i sx_i+1 + h_i sz_i + g ( sx_i sx_i+2 + sz_i sz_i+1 )
	      for (int r=0;r<L;++r) {
		    int rb=r; int rb2=(r+1)%L; int rb3=(r+2)%L;
          if (coupling[r]) { // Ferro convention
            if (config[rb]==config[rb2]) { diag -= (PetscScalar) coupling[r]; } else { diag+= (PetscScalar) coupling[r];} 
           }
        if ((pbc) || ((rb3)>rb)) { // Ferro convention g sx_i sx_i+2
         if (config[rb]==config[rb3]) { diag -= g; } else { diag += g;}
        }
        // now off-diagonal transverse field
        newconfig[rb]=1-config[rb];
        // find this spin-flipped config
        // very likely useless for the full basis ...
        int nleft=0; int nright=0;
		    for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		    for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
		    int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft,nright)]; 
        j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		    if ((j>=Iend) or (j<Istart)) { o_nnz[row_ctr-Istart]++;} else { d_nnz[row_ctr-Istart]++;}
        // now eventually flip the second spin ...
        if ((pbc) || ((rb2)>rb)) {
        newconfig[rb2]=1-config[rb2];
        nleft=0; nright=0;
        for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		    for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
        new_nsa=basis_pointer->particle_sector[std::make_pair(nleft,nright)]; 
        j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		    if ((j>=Iend) or (j<Istart)) { o_nnz[row_ctr-Istart]++;} else { d_nnz[row_ctr-Istart]++;}
        newconfig[rb2]=config[rb2];
        }
        newconfig[rb]=config[rb];
        }

		    // assume always a diagonal part ...
	      if (diag!=0) { d_nnz[row_ctr-Istart]++; }
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
      //  std::cout << "row_ctr=" << row_ctr << " config=";
    //    for (int r=0;r<L;++r) { cout << config[r];  } cout << endl;
	      std::vector<unsigned short int> newconfig=config;
	      std::vector<unsigned short int> newconfA=confA;
	      std::vector<unsigned short int> newconfB=confB;
	      int j;
	      double diag=0.;

        for (int r=0;r<L;++r) {
		    int rb=r; int rb2=(r+1)%L; int rb3=(r+2)%L;
          if (coupling[r]) { // Ferro convention
            if (config[rb]==config[rb2]) { diag -= (PetscScalar) coupling[r]; } else { diag+= (PetscScalar) coupling[r];} 
           }
        if ((pbc) || ((rb3)>rb)) { // Ferro convention g sx_i sx_i+2
         if (config[rb]==config[rb3]) { diag -= g; } else { diag += g;}
        }
        // now off-diagonal transverse field
        newconfig[rb]=1-config[rb];
        // find this spin-flipped config
        // very likely useless for the full basis ...
        int nleft=0; int nright=0;
		    for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		    for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
		    int new_nsa=basis_pointer->particle_sector[std::make_pair(nleft,nright)]; 
        j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		    MatSetValue(*pointer_to_H, row_ctr, j, (PetscScalar) -field[r], ADD_VALUES );
        // now eventually flip the second spin ...
        if ((pbc) || ((rb2)>rb)) {
        newconfig[rb2]=1-config[rb2];
        nleft=0; nright=0;
        for (int p=0;p<LA;++p) { newconfA[p]=newconfig[p]; nleft+=newconfig[p];}
		    for (int p=0;p<LB;++p) { newconfB[p]=newconfig[p+LA]; nright+=newconfig[p+LA];}
        new_nsa=basis_pointer->particle_sector[std::make_pair(nleft,nright)]; 
        j=basis_pointer->starting_conf[new_nsa]+basis_pointer->InverseMapA[new_nsa][newconfA]*basis_pointer->Confs_in_B[new_nsa].size()+basis_pointer->InverseMapB[new_nsa][newconfB];
		    MatSetValue(*pointer_to_H, row_ctr, j, (PetscScalar) -g, ADD_VALUES);
        newconfig[rb2]=config[rb2];
        }
        newconfig[rb]=config[rb];
        }

		    // assume always a diagonal part ...
	      if (diag!=0) { MatSetValue(*pointer_to_H, row_ctr, row_ctr, (PetscScalar) diag, ADD_VALUES ); }
	      }
	      row_ctr++;
	    } // loop over cb
	  } // over ca
	} // over nsA sectors
}


#endif
