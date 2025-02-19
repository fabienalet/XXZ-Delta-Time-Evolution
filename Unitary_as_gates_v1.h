#ifndef FLOQUET_GATES_U
#define FLOQUET_GATES_U

// #define DEBUG
//#include <slepceps.h>
#include <petscmath.h>
#include <vector>
#include <tuple>
#include <bitset>
/*
---------------------------------------
U = U_{+} U_{2} U_{-} U_{2}
U_{+} = R_y ( \pi/2 + \delta_{+}) R_z ( 2\theta) R_y (-(\pi/2 + \delta_{+})))
and
U_{-} = R_y ( \pi/2 - \delta_{-}) R_z ( 2\theta + \epsilon) R_y (-(\pi/2 - \delta_{-})))
where \delta_{\pm} = \delta.
and U_{2} = \exp \left( -i \sum_j \sigma^z_j \sigma^z_{j+1} \right)
and R_y, R_z are rotations about y and z axis respectively.
------------------------------------------
  */

struct MatrixContext {
  PetscInt Lchain;
  Vec All_Rz_theta;
  Vec All_Rz_theta_epsilon;
  Vec Ising_gate;
  std::vector<Mat> Ry_plus_first_gates;
  std::vector<Mat> Ry_plus_second_gates;
  std::vector<Mat> Ry_minus_first_gates;
  std::vector<Mat> Ry_minus_second_gates;
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

/*
---------------------------------------
U = U_{+} U_{2} U_{-} U_{2}
U_{+} = R_y ( \pi/2 + \delta_{+}) R_z ( 2\theta) R_y (-(\pi/2 + \delta_{+})))
and
U_{-} = R_y ( \pi/2 - \delta_{-}) R_z ( 2\theta + \epsilon) R_y (-(\pi/2 - \delta_{-})))
where \delta_{\pm} = \delta.
and U_{2} = \exp \left( -i \sum_j \sigma^z_j \sigma^z_{j+1} \right)
and R_y, R_z are rotations about y and z axis respectively.
------------------------------------------
  */
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
  
  // assume vector x is in the sigma_z basis
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
  PetscErrorCode set_Ry_values(int r);
  PetscErrorCode change_disorder();
  
  int _Istart;
  int _Iend;

  Mat _U;
  MatrixContext *_CTX;

  int Lchain_;
  unsigned long int nconf;

  PetscReal epsilon_;
  PetscReal delta_;
  PetscReal delta_plus_;
  PetscReal delta_minus_;
  PetscReal theta_;
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
  
  Lchain_=12;
  PetscOptionsGetInt(NULL, NULL, "-L", &Lchain_, NULL);
  J_=1.;
  PetscOptionsGetReal(NULL, NULL, "-J", &J_, NULL);

  delta_=0.8;
  PetscOptionsGetReal(NULL, NULL, "-delta", &delta_, NULL);
  delta_plus_=delta_;
  PetscOptionsGetReal(NULL, NULL, "-deltaplus", &deltaplus_, NULL);
  delta_minus_=delta_;
  PetscOptionsGetReal(NULL, NULL, "-deltaminus", &deltaminus_, NULL);
  
  epsilon_=0.05;
  PetscOptionsGetReal(NULL, NULL, "-epsilon", &epsilon_, NULL);
  
  theta_=0.2;
  PetscOptionsGetReal(NULL, NULL, "-theta", &theta_, NULL);
  
  
  Tover2_=1.;
  PetscOptionsGetReal(NULL, NULL, "-Tover2", &Tover2_, NULL);
  T_=2*Tover2_;
  PetscBool T_defined=PETSC_FALSE;
  PetscOptionsGetReal(NULL, NULL, "-T", &T_, &T_defined);
  if (T_defined) { Tover2_=0.5*T_;}

  PetscBool pbc=PETSC_TRUE;
  PetscOptionsGetBool(NULL, NULL, "-pbc", &pbc, NULL);
  
  if (myrank == 0) {
    if (delta_plus_!=delta_) { std::cout << "delta+ = " << delta_plus_ << "\n"; }
    if (delta_minus_!=delta_) { std::cout << "delta- = " << delta_minus_ << "\n"; }
    if ((delta_minus_==delta_) && (delta_plus_==delta_)) { std::cout << "delta = " << delta_ << "\n"; }
    std::cout << "theta = " << theta_ << "\n";
    std::cout << "epsilon = " << epsilon_ << "\n";
    std::cout << "T/2 = " << Tover2_ << "\n";
    
    std::cout << "# J_bonds= {";
    for (int p = 0; p < Lchain_; ++p) { cout << J_coupling[p] << " "; }
    std::cout << " }\n";
  }
}

Unitary_as_gates::~Unitary_as_gates()
{
  for (int i=0;i<Lchain_;i++)
  {
    MatDestroy(&_CTX->Ry_plus_first_gates[i]);
    MatDestroy(&_CTX->Ry_plus_second_gates[i]);
    MatDestroy(&_CTX->Ry_minus_first_gates[i]);
    MatDestroy(&_CTX->Ry_minus_second_gates[i]);
  }
  VecDestroy(&_CTX->All_Rz_theta);
  VecDestroy(&_CTX->All_Rz_theta_epsilon);
  VecDestroy(&_CTX->Ising_gates);
  PetscFree(_CTX);
}

PetscErrorCode Unitary_as_gates::init()
{
  PetscErrorCode ierr;
  
  _CTX->Ry_plus_first_gates.resize(Lchain_);
  _CTX->Ry_plus_second_gates.resize(Lchain_);
  _CTX->Ry_minus_first_gates.resize(Lchain_);
  _CTX->Ry_minus_second_gates.resize(Lchain_);
  
  if (myrank==0) std::cout << "Creating gates ...\n";
  for (int r=0;r<Lchain_;r++)
  { // create gate r
    MatCreate(PETSC_COMM_WORLD,&_CTX->Ry_plus_first_gates[r]);
    MatSetSizes(_CTX->Ry_plus_first_gates[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->Ry_plus_first_gates[r],MATMPIAIJ);
    MatCreate(PETSC_COMM_WORLD,&_CTX->Ry_plus_second_gates[r]);
    MatSetSizes(_CTX->Ry_plus_second_gates[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->Ry_plus_second_gates[r],MATMPIAIJ);
    MatCreate(PETSC_COMM_WORLD,&_CTX->Ry_minus_first_gates[r]);
    MatSetSizes(_CTX->Ry_minus_first_gates[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->Ry_minus_first_gates[r],MATMPIAIJ);
    MatCreate(PETSC_COMM_WORLD,&_CTX->Ry_minus_second_gates[r]);
    MatSetSizes(_CTX->Ry_minus_second_gates[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->Ry_minus_second_gates[r],MATMPIAIJ);
    // preallocate blocks
    {
    std::vector<PetscInt> d_nnz; // list of diagonal (ie internal to the current block) elements
    std::vector<PetscInt> o_nnz; // list of off-diagonal elements
    std::tie(d_nnz, o_nnz) = count_nonzeros(r);
       
    MatMPIAIJSetPreallocation(_CTX->Ry_plus_first[r],0,d_nnz.data(),0,o_nnz.data());
    MatMPIAIJSetPreallocation(_CTX->Ry_plus_second[r],0,d_nnz.data(),0,o_nnz.data());
    MatMPIAIJSetPreallocation(_CTX->Ry_minus_first[r],0,d_nnz.data(),0,o_nnz.data());
    MatMPIAIJSetPreallocation(_CTX->Ry_minus_second[r],0,d_nnz.data(),0,o_nnz.data());
   
    // fill the matrix 
    set_Ry_values(r);
    
    // set properties of the matrix

    MatSetOption(_CTX->Ry_plus_first[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_plus_first[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->Ry_plus_first[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_plus_first[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    ierr=MatAssemblyBegin(_CTX->Ry_plus_first[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr=MatAssemblyEnd(_CTX->Ry_plus_first[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    MatSetOption(_CTX->Ry_plus_second[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_plus_second[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->Ry_plus_second[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_plus_second[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    ierr=MatAssemblyBegin(_CTX->Ry_plus_second[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr=MatAssemblyEnd(_CTX->Ry_plus_second[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    MatSetOption(_CTX->Ry_minus_first[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_minus_first[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->Ry_minus_first[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_minus_first[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    ierr=MatAssemblyBegin(_CTX->Ry_minus_first[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr=MatAssemblyEnd(_CTX->Ry_minus_first[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    MatSetOption(_CTX->Ry_minus_second[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_minus_second[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->Ry_minus_second[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->Ry_minus_second[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    ierr=MatAssemblyBegin(_CTX->Ry_minus_second[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr=MatAssemblyEnd(_CTX->Ry_minus_second[r],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    }
    
    if (myrank==0) std::cout << "Done Creating Ry gates ...\n";

  // Initialize the diagonal operators along x
  MatCreateVecs(Ry_minus_second[0], NULL, &_CTX->All_Rz_theta);
  MatCreateVecs(Ry_minus_second[0], NULL, &_CTX->All_Rz_theta_epsilon);
  MatCreateVecs(Ry_minus_second[0], NULL, &_CTX->Ising_gate);
  //
  for (int i=_Istart;i<_Iend;++i) {
    // bitstring of i = b(i) = 001001 ...
    // count the number of 1's in b(i)
    std::bitset<32> b(i);
    // Diagonal x = exp(-i T/2 ( \sum_j J_j sigma_j^x sigma_j^x )
    double ising_energy=0.;
    double nup=0;
    for (int r=0;r<L;++r) { 
      if (b[r]==b[(r+1+Lchain_)%Lchain_]) { ising_energy+=J_coupling[r];} else { ising_energy-=J_coupling[r];} 
      if (b[r]) { nup++;}
      }
    PetscReal angle_ising=-ising_energy;
    PetscScalar matrix_element=cos(angle_ising)+PETSC_i*sin(angle_ising);
    VecSetValues(_CTX->Ising_gate, 1, &i, &matrix_element, INSERT_VALUES);

    // exp(-i theta) for up

  }
  VecAssemblyBegin(_CTX->All_Rz_theta); VecAssemblyEnd(_CTX->All_Rz_theta);
  VecAssemblyBegin(_CTX->All_Rz_theta_epsilon); VecAssemblyEnd(_CTX->All_Rz_theta_epsilon);
  VecAssemblyBegin(_CTX->Ising_gate); VecAssemblyEnd(_CTX->Ising_gate);
  
  int Lmax=Lchain_; if (!(pbc)) { Lmax=Lchain_-1;}
  // create shell matrices
  ierr=MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nconf,nconf,(void*)_CTX,&_U);CHKERRQ(ierr);
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



PetscErrorCode Unitary_as_gates::set_Ry_values(int r)
{
  // Tover2 ??
  double angle1=PI/2.+delta_plus_;
  double angle2=-angle1;
  double angle3=PI/2.-delta_minus_;
  double angle4=-angle3;
  
  PetscScalar cos1_=cos(angle1/2.);
  PetscScalar cos2_=cos(angle2/2.);
  PetscScalar cos3_=cos(angle3/2.);
  PetscScalar cos4_=cos(angle4/2.);
  PetscScalar sin1_=sin(angle1/2.);
  PetscScalar sin2_=sin(angle2/2.);
  PetscScalar sin3_=sin(angle3/2.);
  PetscScalar sin4_=sin(angle4/2.);
  
  PetscScalar me;
  for (int i=_Istart;i<_Iend;++i) {
    std::bitset<32> b(i);
    MatSetValue(_CTX->Ry_plus_first[r], i, i, cos1_, ADD_VALUES);
    MatSetValue(_CTX->Ry_plus_second[r], i, i, cos2_, ADD_VALUES);
    MatSetValue(_CTX->Ry_minus_first[r], i, i, cos3_, ADD_VALUES);
    MatSetValue(_CTX->Ry_minus_second[r], i, i, cos4_, ADD_VALUES);

    b.flip(r);
    int j = (int)(b.to_ulong());
    if (b[r]) { // don't forget b was flipped, so it's mean b[r] was 0
    // TODO TOCHECK
    MatSetValue(_CTX->Ry_plus_first[r], i, j, sin1_, ADD_VALUES);
    MatSetValue(_CTX->Ry_plus_second[r], i, j, sin2_, ADD_VALUES);
    MatSetValue(_CTX->Ry_minus_first[r], i, j, sin3_, ADD_VALUES);
    MatSetValue(_CTX->Ry_minus_second[r], i, j, sin4_, ADD_VALUES);
    }
    else {
    MatSetValue(_CTX->Ry_plus_first[r], i, j, -sin1_, ADD_VALUES);
    MatSetValue(_CTX->Ry_plus_second[r], i, j, -sin2_, ADD_VALUES);
    MatSetValue(_CTX->Ry_minus_first[r], i, j, -sin3_, ADD_VALUES);
    MatSetValue(_CTX->Ry_minus_second[r], i, j, -sin4_, ADD_VALUES);
    }
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





