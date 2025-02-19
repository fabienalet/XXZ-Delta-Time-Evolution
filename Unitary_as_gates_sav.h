#ifndef FLOQUET_GATES_U
#define FLOQUET_GATES_U

// #define DEBUG
//#include <slepceps.h>
#include <petscmath.h>
#include <vector>
#include <tuple>
#include <bitset>


struct MatrixContext {
  PetscInt Lchain;
  Vec 2qubits_gates_x;
  Vec 2qubits_gates_x_inv;
  std::vector<Mat> 1qubit_gates;
  std::vector<Mat> 1qubit_gates_inv;
};

// useless
PetscErrorCode VecMultScalar(Vec x, PetscScalar *scalar)
{
  /*
  x[i] <- x[i] * scalar[i]
  /!\ assuming x and scalar have the same size
  */
  PetscErrorCode    ierr;
  PetscScalar       *px;
  PetscInt          Istart,Iend;
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&Istart,&Iend);CHKERRQ(ierr);
  for (int i=0;i<Iend-Istart;i++)
  {
    px[i] *= scalar[i+Istart];
  }
  VecRestoreArray(x,&px);
}


PetscErrorCode MatMultU(Mat M,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatrixContext     *ctx;
 // Vec               x2;
  //Vec               x3;
  // First executable line of user provided PETSc routine
  PetscFunctionBeginUser;
  VecSet(y,0.);
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
 // std::cout << "MatMult U : step 0\n";
 // VecView(x,PETSC_VIEWER_STDOUT_WORLD); 
  
  // assume vector x is in the sigma_x basis
  // first apply all 1-qubit gates (U_plus)
  for (int i=0;i<ctx->Lchain;++i) {
  MatMult(ctx->1qubit_gates[i], x, y);
  if (i < ctx->Lchain - 1) { VecSwap(y, x); }
  }
  // result is in y
  // std::cout << "MatMult U : step 1\n";
 // VecView(y,PETSC_VIEWER_STDOUT_WORLD); 
  // Then apply 2 qubits-gate U_2 (diagonal operation in sigma_x basis)
  VecPointwiseMult(x, y, ctx->2qubits_gates_x);
  // result is in x now
  // first apply inverse of all 1-qubit gates (U_minus)
  for (int i=0;i<ctx->Lchain;++i) {
  MatMult(ctx->1qubit_gates_inv[i], x, y);
  VecSwap(y, x);
  }
  // result is in x now
  // VecView(x,PETSC_VIEWER_STDOUT_WORLD); 
  // Finally apply 2 qubits-gate U_2_inv (diagonal operation in sigma_x basis)
  VecPointwiseMult(y, x, ctx->2qubits_gates_x_inv);

PetscFunctionReturn(0);
}

class Unitary_as_gates
{
  //typedef std::vector<unsigned short int> Conf;
public:
  Unitary_as_gates(int _myrank, int mpisize); // no basis pointer, assume full 2^L basis
  ~Unitary_as_gates();
  PetscErrorCode init();
  std::tuple<std::vector<PetscInt>, std::vector<PetscInt>> count_nonzeros(int r);
  PetscErrorCode set_matrix_values(int r);
  PetscErrorCode change_disorder();
  
  int _Istart;
  int _Iend;

  Mat _U;
  MatrixContext *_CTX;

  int Lchain_;
  unsigned long int nconf;

  PetscReal hx_;
  PetscReal Gamma_;
  PetscReal J_;
  std::vector<PetscReal> J_coupling;
  
  PetscReal Tover2_;
  PetscReal T_;

  PetscBool pbc;
  int myrank;

  void get_parameters();

private:
};

Unitary_as_gates::Unitary_as_gates(int _myrank, int mpisize)
{
  myrank = _myrank;

  get_parameters();

  nconf = pow(2,Lchain_);
  PetscNew(&_CTX);

  _CTX->Lchain=Lchain_;
  
  std::vector<int> local_block_sizes(mpisize,nconf/mpisize);
  for(size_t i=0; i< nconf%mpisize; i++) local_block_sizes[i]++; // distribute evenly
  _Istart = 0;
  for (size_t i=0; i<myrank; i++) _Istart += local_block_sizes[i];
  _Iend = _Istart + local_block_sizes[myrank];

  init();

}

void Unitary_as_gates::get_parameters() {
  
  Lchain_=6;
  PetscOptionsGetInt(NULL, NULL, "-L", &Lchain_, NULL);
  J_=1.;
  PetscOptionsGetReal(NULL, NULL, "-J", &J_, NULL);
  hx_=1.2;
  PetscOptionsGetReal(NULL, NULL, "-hx", &hx_, NULL);
  Gamma_=1.4;
  PetscOptionsGetReal(NULL, NULL, "-Gamma", &Gamma_, NULL);
  PetscOptionsGetReal(NULL, NULL, "-gamma", &Gamma_, NULL);
  
  Tover2_=1.;
  PetscOptionsGetReal(NULL, NULL, "-Tover2", &Tover2_, NULL);
  T_=2*Tover2_;
  PetscBool T_defined=PETSC_FALSE;
  PetscOptionsGetReal(NULL, NULL, "-T", &T_, &T_defined);
  if (T_defined) { Tover2_=0.5*T_;}

  PetscBool pbc=PETSC_TRUE;
  PetscOptionsGetBool(NULL, NULL, "-pbc", &pbc, NULL);
  
  if (myrank == 0) {
    std::cout << "Gamma = " << Gamma_ << "\n";
    std::cout << "hx = " << hx_ << "\n";
    std::cout << "T/2 = " << Tover2_ << "\n";
    // ...
    std::cout << "# J_bonds= {";
    for (int p = 0; p < Lchain_; ++p) { cout << J_coupling[p] << " "; }
    std::cout << " }\n";
  }
}

  

Unitary_as_gates::~Unitary_as_gates()
{
  for (int i=0;i<Lchain_;i++)
  {
    MatDestroy(&_CTX->1qubit_gates[i]);
    MatDestroy(&_CTX->1qubit_gates_inv[i]);
  }
  VecDestroy(&_CTX->2qubits_gates_x);
  VecDestroy(&_CTX->2qubits_gates_x_inv);
  PetscFree(_CTX);
}

PetscErrorCode Unitary_as_gates::init()
{
  PetscErrorCode ierr;
  
  _CTX->1qubit_gates.resize(Lchain_);
  _CTX->1qubit_gates_inv.resize(Lchain_);
  
  if (myrank==0) std::cout << "Creating gates ...\n";
  for (int r=0;r<Lchain_;r++)
  { // create gate r
    MatCreate(PETSC_COMM_WORLD,&_CTX->1qubit_gates[r]);
    MatSetSizes(_CTX->1qubit_gates[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->1qubit_gates[r],MATMPIAIJ);
    MatCreate(PETSC_COMM_WORLD,&_CTX->1qubit_gates_inv[r]);
    MatSetSizes(_CTX->1qubit_gates_inv[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->1qubit_gates_inv[r],MATMPIAIJ);
    
    // preallocate blocks
    {
    std::vector<PetscInt> d_nnz; // list of diagonal (ie internal to the current block) elements
    std::vector<PetscInt> o_nnz; // list of off-diagonal elements
    std::tie(d_nnz, o_nnz) = count_nonzeros(r);
       
    MatMPIAIJSetPreallocation(_CTX->1qubit_gates[r],0,d_nnz.data(),0,o_nnz.data());
    MatMPIAIJSetPreallocation(_CTX->1qubit_gates_inv[r],0,d_nnz.data(),0,o_nnz.data());
   
    // fill the matrix 
    set_matrix_values(r,0);
    
    // set properties of the matrix
    MatSetOption(_CTX->1qubit_gates[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->1qubit_gates[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->1qubit_gates[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->1qubit_gates[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    ierr=MatAssemblyBegin(_CTX->1qubit_gates[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr=MatAssemblyEnd(_CTX->1qubit_gates[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    set_matrix_values(r,1);
    // set properties of the matrix
    MatSetOption(_CTX->1qubit_gates_inv[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->1qubit_gates_inv[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->1qubit_gates_inv[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->1qubit_gates_inv[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    ierr=MatAssemblyBegin(_CTX->1qubit_gates_inv[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr=MatAssemblyEnd(_CTX->1qubit_gates_inv[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    }
    
   // std::cout << "Assembly done\n";
  //   MatView(_CTX->Hadamard_gates[r],PETSC_VIEWER_STDOUT_WORLD);
  }
    if (myrank==0) std::cout << "Done Creating gates ...\n";

  // Initialize the diagonal operators along x
  MatCreateVecs(_1qubit_gates[0], NULL, &_CTX->2qubits_gates_x);
  MatCreateVecs(_1qubit_gates[0], NULL, &_CTX->2qubits_gates_x_inv);
  //
  for (int i=_Istart;i<_Iend;++i) {
    // bitstring of i = b(i) = 001001 ...
    // count the number of 1's in b(i)
    std::bitset<32> b(i);
    // Diagonal x = exp(-i T/2 ( \sum_j J_j sigma_j^x sigma_j^x )
    double ising_energy_x=0.;
    for (int r=0;r<L;++r) { if (b[r]==b[(r+1+Lchain_)%Lchain_]) { ising_energy_x+=J_coupling[r];} else { ising_energy_x-=J_coupling[r];} }
    
    PetscReal angle=-Tover2_*ising_energy_x;

    PetscScalar matrix_element=cos(angle)+PETSC_i*sin(angle);
    PetscScalar matrix_element_inv=cos(angle)-PETSC_i*sin(angle);
    VecSetValues(_CTX->2qubits_gates_x, 1, &i, &matrix_element, INSERT_VALUES);
    VecSetValues(_CTX->2qubits_gates_x_inv, 1, &i, &matrix_element_inv, INSERT_VALUES);
  }
  VecAssemblyBegin(_CTX->2qubits_gates_x); VecAssemblyEnd(_CTX->2qubits_gates_x);
  VecAssemblyBegin(_CTX->2qubits_gates_x_inv); VecAssemblyEnd(_CTX->2qubits_gates_x_inv);
  
  // Initialize the diagonal operators along x
  MatCreateVecs(_CTX->Hadamard_gates[0], NULL, &_CTX->Diagonal_Unitary_x);
  
  int Lmax=Lchain_; if (!(pbc)) { Lmax=Lchain_-1;}
//std::cout << "J matrix element :\n";
 // for (int r=0;r<Lchain_;++r) { cout << (h_+g_*sqrt(1-Gamma_*Gamma_)*G_coupling[r]) << endl;}
  for (int i=_Istart;i<_Iend;++i) {
    // bitstring of i = b(i) = 001001 ...
    // count the number of 1's in b(i)
    std::bitset<32> b(i);
   
    PetscReal me=0.;
    for (int r=0;r<Lmax;++r) {
      int r2=r; int rb=r+1;
     // if (other_convention) { r2=Lchain_-r; rb=Lchain_-r-1;}
      if (b[r2]==b[(rb+Lchain_)%Lchain_]) { me+=b_;} else { me-=b_;}
     // if (b[r2]) { me+=(h_+g_*sqrt(1-Gamma_*Gamma_)*G_coupling[r]);} else { me-=(h_+g_*sqrt(1-Gamma_*Gamma_)*G_coupling[r]); }
    }
    
    for (int r=0;r<Lchain_;++r) {
       if (b[r]) { me+=(h_+g_*sqrt(1-Gamma_*Gamma_)*G_coupling[r]);} else { me-=(h_+g_*sqrt(1-Gamma_*Gamma_)*G_coupling[r]); }
    }
    PetscReal angle=-tau_*me;
    PetscScalar matrix_element=cos(angle)+PETSC_i*sin(angle);
    VecSetValues(_CTX->Diagonal_Unitary_x, 1, &i, &matrix_element, INSERT_VALUES);
  }
  VecAssemblyBegin(_CTX->Diagonal_Unitary_x); VecAssemblyEnd(_CTX->Diagonal_Unitary_x);

  // create shell matrices
  ierr=MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nconf,nconf,(void*)_CTX,&_U);CHKERRQ(ierr);
  ierr=MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nconf,nconf,(void*)_CTX,&_Hadamard);CHKERRQ(ierr);
  // define multiplication operations
  ierr=MatShellSetOperation(_U,MATOP_MULT,(void(*)())MatMultU);CHKERRQ(ierr);
  // declare matrix to be Hermitian
  MatSetOption(_U,	MAT_SYMMETRIC, PETSC_TRUE);
  MatSetOption(_U, MAT_SYMMETRY_ETERNAL, PETSC_TRUE);
  //std::cout << "Diagonal x: \n";
  //VecView(_CTX->Diagonal_Unitary_x,PETSC_VIEWER_STDOUT_WORLD); 
  //std::cout << "Diagonal z: \n";
  //VecView(_CTX->Diagonal_Unitary_z,PETSC_VIEWER_STDOUT_WORLD); 
  for (int r=0;r<Lchain_;++r) {
  //  cout << "Gate " << r << "\n";
 // MatView(_CTX->Hadamard_gates[r], PETSC_VIEWER_STDOUT_WORLD);
  }
}



PetscErrorCode Unitary_as_gates::set_matrix_values(int r,bool inverse)
{
/*** A = 1 qubit gate = (h_x sigma_x + gamma sigma_z ) */  
//  exp(-i A T/2 ) =  ( cos( R T / 2  ) - i sin(R T / 2 )/ R            i Gamma sin(RT/2) / R       )
//                    ( i Gamma sin(RT/2) / R                          cos( R T / 2  ) + i sin(R T / 2 )/ R )
// with R = sqrt ( hx^2 + Gamma^2 )
  PetscReal R = sqrt ( hx_*hx_ + Gamma*Gamma );
  PetscReal angle= R*Tover2_;
  PetscScalar diag_11=cos(angle)-PETSC_i*sin(angle)/R;
  PetscScalar diag_00=cos(angle)+PETSC_i*sin(angle)/R;
  PetscScalar off_diag=-Gamma_PETSC_i*sin(angle);
  PetscScalar me;
  for (int i=_Istart;i<_Iend;++i) {
    std::bitset<32> b(i);

    if (inverse) { if (b[r]) { me=diag_00; } else { me=diag_11;} 
    MatSetValue(_CTX->1qubit_gates[r], i, i, me, ADD_VALUES);
    }
    else { if (b[r]) { me=diag_11; } else { me=diag_00;} 
     MatSetValue(_CTX->1qubit_gates[r], i, i, XXX, ADD_VALUES); }

    b.flip(r);
    int j = (int)(b.to_ulong());
    if (inverse) { MatSetValue(_CTX->1qubit_gates_inv[r], i, j, -off_diag, ADD_VALUES); }
    else { MatSetValue(_CTX->1qubit_gates[r], i, j, off_diag, ADD_VALUES); }
  }
  
}

std::tuple<std::vector<PetscInt>, std::vector<PetscInt>> Unitary_as_gates::count_nonzeros(int r)
{
  std::vector<PetscInt> d_nnz (_Iend-_Istart, 0); // list of diagonal (ie internal to the current block) elements
  std::vector<PetscInt> o_nnz (_Iend-_Istart, 0); // list of off-diagonal elements

  for (int i=_Istart;i<_Iend;++i) {
    // there will be one diagonal element for sure
      d_nnz[i - _Istart]++;
    // now do the bit flip on r
    std::bitset<32> b(i);
    int r2=r;
      if (other_convention) { r2=Lchain_-r; }

    b.flip(r2);
    int j = (int)(b.to_ulong());
    if ((j >= _Iend) or (j < _Istart)) { o_nnz[i - _Istart]++; }
      else { d_nnz[i - _Istart]++;}
  }

  return std::make_tuple(d_nnz, o_nnz);
}

#endif





