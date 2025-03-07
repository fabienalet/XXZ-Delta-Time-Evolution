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
  PetscReal theta_;
  PetscReal epsilon_;
  PetscReal delta_plus_;
  PetscReal delta_minus_;
  Vec Ising_gate;
  std::vector<Mat> U_plus_gates;
  std::vector<Mat> U_minus_gates;
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


PetscErrorCode MatMultUplus(Mat M,int r,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatrixContext     *ctx;
  const PetscScalar *xloc;
  PetscInt lo,hi;
  // First executable line of user provided PETSc routine
  PetscFunctionBeginUser;

  MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  VecGetOwnershipRange(x, &lo, &hi);

  PetscCall(VecGetArrayRead(x, &xloc));
  PetscCall(VecSet(y, 0.));

  PetscReal costp=cos(ctx->theta_);
  PetscReal sintp=sin(ctx->theta_);
  PetscReal phi_plus=PETSC_PI/2.+ctx->delta_plus_;
  PetscScalar valii1=costp+PETSC_i*sintp*cos(PETSC_PI-phi_plus);
  PetscScalar valii2=costp-PETSC_i*sintp*cos(PETSC_PI-phi_plus);
  PetscScalar valij=PETSC_i*sintp*sin(PETSC_PI-phi_plus);
  //cout << "plus : " << valii1 << " " << valii2 << " " << valij << endl;
  /*
  PetscReal costm=cos(ctx->theta_+ctx->epsilon_);
  PetscReal sintm=sin(ctx->theta_+ctx->epsilon_);
  PetscScalar valii=costm+PETSC_i*sintp*cos(PETSC_PI-phi_minus);
  PetscScalar valij=PETSC_i*sintm*sin(PETSC_PI-phi_minus);
  */
 PetscScalar mi,mj,xl;
  for (int i=lo;i<hi;++i) {
    std::bitset<32> b(i);
    b.flip(r);
    int j = (int)(b.to_ulong());
    b.flip(r);
    xl=xloc[i-lo];
    // maybe don't flip again and reverse the if ...
    if (b[r]) {  mi=xl*valii1; VecSetValues(y, 1, &i, &mi, ADD_VALUES);}
    else { mi=xl*valii2; VecSetValues(y, 1, &i, &mi, ADD_VALUES);}
    mj=xl*valij;
      VecSetValues(y, 1, &j, &mj, ADD_VALUES);
    }

  VecRestoreArrayRead(x, &xloc);
  VecAssemblyBegin(y);
  VecAssemblyEnd(y);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultUminus(Mat M,int r,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatrixContext     *ctx;
  const PetscScalar *xloc;
  PetscInt lo,hi;
  // First executable line of user provided PETSc routine
  PetscFunctionBeginUser;
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
 // cout << "Viewing x\n"; VecView(x,PETSC_VIEWER_STDOUT_WORLD); 
  MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  VecGetOwnershipRange(x, &lo, &hi);

  PetscCall(VecGetArrayRead(x, &xloc));
  PetscCall(VecSet(y, 0.));

  PetscReal phi_minus=PETSC_PI/2.-ctx->delta_minus_;
  
  PetscReal costm=cos(ctx->theta_+ctx->epsilon_);
  PetscReal sintm=sin(ctx->theta_+ctx->epsilon_);
  PetscScalar valii1=costm+PETSC_i*sintm*cos(PETSC_PI-phi_minus);
  PetscScalar valii2=costm-PETSC_i*sintm*cos(PETSC_PI-phi_minus);
  PetscScalar valij=PETSC_i*sintm*sin(PETSC_PI-phi_minus);
 // cout << "minus : " << valii1 << " " << valii2 << " " << valij << endl;
 PetscScalar mi,mj,xl;
  for (int i=lo;i<hi;++i) {
  //  if (myrank==1) cout << "rank=" << myrank << " i=" << i << " xloc[i]=" << xloc[i-lo] << endl;
    std::bitset<32> b(i);
    b.flip(r);
    int j = (int)(b.to_ulong());
    b.flip(r);
    xl=xloc[i-lo];
    // maybe don't flip again and reverse the if ...
    if (b[r]) {  mi=xl*valii1; VecSetValues(y, 1, &i, &mi, ADD_VALUES); }
    else { mi=xl*valii2; VecSetValues(y, 1, &i, &mi, ADD_VALUES);}
   // cout << "myrank " << myrank << " is Adding to y : " << i << " " << mi << endl;
    mj=xl*valij;
      VecSetValues(y, 1, &j, &mj, ADD_VALUES);
   //   cout << "myrank " << myrank << " is Adding to y : " << j << " " << mj << endl;
    }

  VecRestoreArrayRead(x, &xloc);
  VecAssemblyBegin(y);
  VecAssemblyEnd(y);
  //cout << "Viewing y\n"; VecView(y,PETSC_VIEWER_STDOUT_WORLD); 
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultU3(Mat M,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatrixContext     *ctx;
  Vec               x2;
  // First executable line of user provided PETSc routine
  PetscFunctionBeginUser;
  VecSet(y,0.);
  VecDuplicate(x,&x2);
  VecCopy(x,x2);
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
 // std::cout << "MatMult U : step 0\n";
 // apply 2 qubits-gate U_2 (diagonal operation in sigma_z basis)
  VecPointwiseMult(y, x, ctx->Ising_gate);
  // result in y
  // first apply all 1-qubit gates (U_minus)
  for (int i=0;i<ctx->Lchain;++i) {
  MatMultUminus(M,i, y, x2);
  if (i < (ctx->Lchain - 1)) { VecSwap(y, x2); }
  }
 
  // result is in x2
  // apply 2 qubits-gate U_2 (diagonal operation in sigma_z basis)
  VecPointwiseMult(y, x2, ctx->Ising_gate);
  // result in y

 // Apply all 1-qubit gates (U_plus)
 for (int i=0;i<ctx->Lchain;++i) {
  MatMultUplus(M,i, y, x2);
  if (i < (ctx->Lchain - 1)) { VecSwap(y, x2); }
  }

  // result is in x2
  // Do a final swap
  VecSwap(x2, y);
  
PetscFunctionReturn(0);
}

PetscErrorCode MatMultU2(Mat M,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatrixContext     *ctx;
  Vec               x2;
  //Vec               x3;
  // First executable line of user provided PETSc routine
  PetscFunctionBeginUser;
  VecSet(y,0.);
  VecDuplicate(x,&x2);
  VecCopy(x,x2);
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
 // std::cout << "MatMult U : step 0\n";
 // apply 2 qubits-gate U_2 (diagonal operation in sigma_z basis)
  VecPointwiseMult(y, x2, ctx->Ising_gate);
  // result in y
  // first apply all 1-qubit gates (U_minus)
  for (int i=0;i<ctx->Lchain;++i) {
  MatMult(ctx->U_minus_gates[i], y, x2);
  if (i < (ctx->Lchain - 1)) { VecSwap(y, x2); }
  }
 
  // result is in x2
  // apply 2 qubits-gate U_2 (diagonal operation in sigma_z basis)
  VecPointwiseMult(y, x2, ctx->Ising_gate);
  // result in y

 // Apply all 1-qubit gates (U_plus)
  for (int i=0;i<ctx->Lchain;++i) {
  MatMult(ctx->U_plus_gates[i], y, x2);
  if (i < (ctx->Lchain - 1)) { VecSwap(y, x2); }
  }
  // result is in x2
  // Do a final swap
  VecSwap(x2, y);
  
PetscFunctionReturn(0);
}



PetscErrorCode MatMultU1(Mat M,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatrixContext     *ctx;
  Vec               x2;
  //Vec               x3;
  // First executable line of user provided PETSc routine
  PetscFunctionBeginUser;
  VecSet(y,0.);
  VecDuplicate(x,&x2);
  VecCopy(x,x2);
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
 // std::cout << "MatMult U : step 0\n";
 // VecView(x,PETSC_VIEWER_STDOUT_WORLD); 
  
  // assume vector x is in the sigma_z basis
  // first apply all 1-qubit gates (U_plus)
  for (int i=0;i<ctx->Lchain;++i) {
  MatMult(ctx->U_minus_gates[i], x2, y);
  if (i < (ctx->Lchain - 1)) { VecSwap(y, x2); }
  }
  // result is in y
   // VecView(y,PETSC_VIEWER_STDOUT_WORLD); 
  // Then apply 2 qubits-gate U_2 (diagonal operation in sigma_z basis)
  VecPointwiseMult(x2, y, ctx->Ising_gate);
  // result is in x now

  // Apply all 1-qubit gates (U_minus)
  for (int i=0;i<ctx->Lchain;++i) {
  MatMult(ctx->U_plus_gates[i], x2, y);
  if (i < (ctx->Lchain - 1)) { VecSwap(y, x2); }
  }
  // result is in y
   // VecView(y,PETSC_VIEWER_STDOUT_WORLD); 
  // Then apply 2 qubits-gate U_2 (diagonal operation in sigma_z basis)
  VecPointwiseMult(x2, y, ctx->Ising_gate);
  // result is in x now

  // Do a final swap
  VecSwap(x2, y);
  
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
  PetscErrorCode set_gate_values(int r);
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

  _CTX->epsilon_=epsilon_;
  _CTX->theta_=theta_;
  _CTX->delta_plus_=delta_plus_;
  _CTX->delta_minus_=delta_minus_;

}

void Unitary_as_gates::get_parameters() {
  
  PetscBool pi_units=PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-pi", &pi_units, NULL);

  Lchain_=12;
  PetscOptionsGetInt(NULL, NULL, "-L", &Lchain_, NULL);
  J_=1.;
  PetscOptionsGetReal(NULL, NULL, "-J", &J_, NULL);
  J_coupling.resize(Lchain_,J_);

  delta_=0.8;
  PetscOptionsGetReal(NULL, NULL, "-delta", &delta_, NULL);
  delta_plus_=delta_;
  PetscOptionsGetReal(NULL, NULL, "-deltaplus", &delta_plus_, NULL);
  delta_minus_=delta_;
  PetscOptionsGetReal(NULL, NULL, "-deltaminus", &delta_minus_, NULL);
  
  epsilon_=0.05;
  PetscOptionsGetReal(NULL, NULL, "-epsilon", &epsilon_, NULL);
  
  theta_=0.2;
  PetscOptionsGetReal(NULL, NULL, "-theta", &theta_, NULL);


  if (pi_units) {
    delta_=delta_*PETSC_PI/2;
    delta_plus_=delta_plus_*PETSC_PI/2;
    delta_minus_=delta_minus_*PETSC_PI/2;
    epsilon_=epsilon_*PETSC_PI;
    theta_=theta_*PETSC_PI;
  }

  
  Tover2_=1.;
  PetscOptionsGetReal(NULL, NULL, "-Tover2", &Tover2_, NULL);
  T_=2*Tover2_;
  PetscBool T_defined=PETSC_FALSE;
  PetscOptionsGetReal(NULL, NULL, "-T", &T_, &T_defined);
  if (T_defined) { Tover2_=0.5*T_;}

  pbc=PETSC_FALSE;
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
  { MatDestroy(&_CTX->U_plus_gates[i]);
    MatDestroy(&_CTX->U_minus_gates[i]); }
  VecDestroy(&_CTX->Ising_gate);
  PetscFree(_CTX);
}

PetscErrorCode Unitary_as_gates::init()
{
  PetscErrorCode ierr;
  PetscBool U3=PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-U3", &U3, NULL);

  if (!(U3)) {
    
  _CTX->U_plus_gates.resize(Lchain_);
  _CTX->U_minus_gates.resize(Lchain_);
  
  for (int r=0;r<Lchain_;r++)
  { // create gate r
    MatCreate(PETSC_COMM_WORLD,&_CTX->U_plus_gates[r]);
    MatSetSizes(_CTX->U_plus_gates[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->U_plus_gates[r],MATMPIAIJ);
    MatCreate(PETSC_COMM_WORLD,&_CTX->U_minus_gates[r]);
    MatSetSizes(_CTX->U_minus_gates[r],PETSC_DECIDE,PETSC_DECIDE,nconf,nconf); // maybe use Istart Iend here?
    MatSetType(_CTX->U_minus_gates[r],MATMPIAIJ);
    // preallocate blocks
    {
    std::vector<PetscInt> d_nnz; // list of diagonal (ie internal to the current block) elements
    std::vector<PetscInt> o_nnz; // list of off-diagonal elements
    std::tie(d_nnz, o_nnz) = count_nonzeros(r);
       
    MatMPIAIJSetPreallocation(_CTX->U_plus_gates[r],0,d_nnz.data(),0,o_nnz.data());
    MatMPIAIJSetPreallocation(_CTX->U_minus_gates[r],0,d_nnz.data(),0,o_nnz.data());
   
    // fill the matrix 
    set_gate_values(r);
    
    // set properties of the matrix

    MatSetOption(_CTX->U_plus_gates[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->U_plus_gates[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->U_plus_gates[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->U_plus_gates[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatAssemblyBegin(_CTX->U_plus_gates[r],MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(_CTX->U_plus_gates[r],MAT_FINAL_ASSEMBLY);

    MatSetOption(_CTX->U_minus_gates[r],MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->U_minus_gates[r], MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatSetOption(_CTX->U_minus_gates[r],MAT_SYMMETRIC,PETSC_TRUE);
    MatSetOption(_CTX->U_minus_gates[r],MAT_SYMMETRY_ETERNAL,PETSC_TRUE);
    MatAssemblyBegin(_CTX->U_minus_gates[r],MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(_CTX->U_minus_gates[r],MAT_FINAL_ASSEMBLY);

    }
  }
    if (myrank==0) std::cout << "Done Creating Uplus-Uminus gates ...\n";
  }

  // create shell matrices
  ierr=MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,nconf,nconf,(void*)_CTX,&_U);CHKERRQ(ierr);
  if (U3) { ierr=MatShellSetOperation(_U,MATOP_MULT,(void(*)())MatMultU3);CHKERRQ(ierr); }
  else { ierr=MatShellSetOperation(_U,MATOP_MULT,(void(*)())MatMultU2);CHKERRQ(ierr);  }
  // declare matrix to be symmetric
  MatSetOption(_U,	MAT_SYMMETRIC, PETSC_TRUE);
  MatSetOption(_U, MAT_SYMMETRY_ETERNAL, PETSC_TRUE);

  MatCreateVecs(_U, NULL, &_CTX->Ising_gate);
  PetscInt Lmax=Lchain_-1;
  if (pbc) { Lmax=Lchain_;}
  for (int i=_Istart;i<_Iend;++i) {
    std::bitset<32> b(i);
    double ising_energy=0.;
    double nup=0;
    
    for (int r=0;r<Lmax;++r) { 
      if (b[r]==b[(r+1+Lchain_)%Lchain_]) { ising_energy+=J_coupling[r];} else { ising_energy-=J_coupling[r];} 
      }
    PetscReal angle_ising=ising_energy;

    PetscScalar matrix_element=cos(angle_ising)+PETSC_i*sin(angle_ising);
    VecSetValues(_CTX->Ising_gate, 1, &i, &matrix_element, INSERT_VALUES);
   // cout << "angle= " << angle_ising << " me=" << matrix_element << endl;
  }
  VecAssemblyBegin(_CTX->Ising_gate); VecAssemblyEnd(_CTX->Ising_gate);

}



PetscErrorCode Unitary_as_gates::set_gate_values(int r)
{
/*
U_{+} = R_y ( \pi/2 + \delta_{+}) R_z ( 2\theta) R_y (-(\pi/2 + \delta_{+})))
and
U_{-} = R_y ( \pi/2 - \delta_{-}) R_z ( 2\theta + \epsilon) R_y (-(\pi/2 - \delta_{-})))
where \delta_{\pm} = \delta.
*/

  PetscReal phi_plus=PETSC_PI/2.+delta_plus_;
 PetscReal phi_minus=PETSC_PI/2.-delta_minus_;
 
  /*
  PetscScalar cos1_=cos(angle1/2.);
  PetscScalar cos2_=cos(angle2/2.);
  PetscScalar cos3_=cos(angle3/2.);
  PetscScalar cos4_=cos(angle4/2.);
  PetscScalar sin1_=sin(angle1/2.);
  PetscScalar sin2_=sin(angle2/2.);
  PetscScalar sin3_=sin(angle3/2.);
  PetscScalar sin4_=sin(angle4/2.);
  
  PetscScalar c1c2=cos1_*cos2_;
  PetscScalar s1s2=sin1_*sin2_;
  PetscScalar c1s2=cos1_*sin2_;
  PetscScalar c2s1=cos2_*sin1_;
  PetscScalar c3c4=cos3_*cos4_;
  PetscScalar s3s4=sin3_*sin4_;
  PetscScalar c3s4=cos3_*sin4_;
  PetscScalar c4s3=cos4_*sin3_;
  */
  
  PetscReal costp=cos(theta_);
  PetscReal sintp=sin(theta_);
  PetscReal costm=cos(theta_+epsilon_); //+epsilon_);
  PetscReal sintm=sin(theta_+epsilon_); //+epsilon_);
  
  
 // cout << "U2 plus : " << costp+PETSC_i*sintp*cos(PETSC_PI-phi_plus) << " " << costp-PETSC_i*sintp*cos(PETSC_PI-phi_plus) << " " << PETSC_i*sintp*sin(PETSC_PI-phi_plus) << endl;
 // cout << "U2 minus : " << costm+PETSC_i*sintm*cos(PETSC_PI-phi_minus) << " " << costm-PETSC_i*sintm*cos(PETSC_PI-phi_minus) << " " << PETSC_i*sintm*sin(PETSC_PI-phi_minus) << endl;
  
  for (int i=_Istart;i<_Iend;++i) {
    std::bitset<32> b(i);
    b.flip(r);
    int j = (int)(b.to_ulong());
    b.flip(r);
    // maybe don't flip again and reverse the if ...
    if (b[r]) {
  //  MatSetValue(_CTX->U_plus_gates[r], i, i, c1c2*(costp-PETSC_i*sintp)-s1s2*(costp+PETSC_i*sintp), ADD_VALUES);
   // MatSetValue(_CTX->U_minus_gates[r], i, i, c3c4*(costm-PETSC_i*sintm)-s3s4*(costm+PETSC_i*sintm), ADD_VALUES);
    // Asmi's notes

     MatSetValue(_CTX->U_plus_gates[r], i, i, costp+PETSC_i*sintp*cos(PETSC_PI-phi_plus), INSERT_VALUES);
     MatSetValue(_CTX->U_minus_gates[r], i, i, costm+PETSC_i*sintm*cos(PETSC_PI-phi_minus), INSERT_VALUES);
      cout << "U+_11=" << costp+PETSC_i*sintp*cos(PETSC_PI-phi_plus) << endl;
      cout << "U-_11=" << costm+PETSC_i*sintm*cos(PETSC_PI-phi_minus) << endl;
   // MatSetValue(_CTX->U_plus_gates[r], i, j, -c1s2*(costp-PETSC_i*sintp)-c2s1*(costp+PETSC_i*sintp), ADD_VALUES);
   // MatSetValue(_CTX->U_minus_gates[r], i, j, -c3s4*(costp-PETSC_i*sintm)-c4s3*(costp+PETSC_i*sintp), ADD_VALUES);
   // Asmi's notes
    MatSetValue(_CTX->U_plus_gates[r], i, j, -PETSC_i*sintp*sin(PETSC_PI-phi_plus), INSERT_VALUES);
    cout << "U+_10=" << -PETSC_i*sintp*sin(PETSC_PI-phi_plus) << endl;
   MatSetValue(_CTX->U_minus_gates[r], i, j, -PETSC_i*sintm*sin(PETSC_PI-phi_minus), INSERT_VALUES);
   cout << "U-_10=" << -PETSC_i*sintm*sin(PETSC_PI-phi_minus) << endl;
    
    }

    else {
   //   MatSetValue(_CTX->U_plus_gates[r], i, i, -s1s2*(costp-PETSC_i*sintp)+c1c2*(costp+PETSC_i*sintp), ADD_VALUES);
   //   MatSetValue(_CTX->U_minus_gates[r], i, i, -s3s4*(costm-PETSC_i*sintm)+c1c2*(costm+PETSC_i*sintm), ADD_VALUES);
      // Asmi's notes

     MatSetValue(_CTX->U_plus_gates[r], i, i, costp-PETSC_i*sintp*cos(PETSC_PI-phi_plus), INSERT_VALUES);
     cout << "U+_00=" << costp-PETSC_i*sintp*cos(PETSC_PI-phi_plus) << endl;
     MatSetValue(_CTX->U_minus_gates[r], i, i, costm-PETSC_i*sintm*cos(PETSC_PI-phi_minus), INSERT_VALUES);
     cout << "U-_00=" << costm-PETSC_i*sintm*cos(PETSC_PI-phi_minus) << endl;
    //  MatSetValue(_CTX->U_plus_gates[r], i, j, c2s1*(costp-PETSC_i*sintp)+c1s2*(costp+PETSC_i*sintp), ADD_VALUES);
     // MatSetValue(_CTX->U_minus_gates[r], i, j, c4s3*(costp-PETSC_i*sintp)+c3s4*(costp+PETSC_i*sintp), ADD_VALUES);
      // Asmi's notes
    MatSetValue(_CTX->U_plus_gates[r], i, j, -PETSC_i*sintp*sin(PETSC_PI-phi_plus), INSERT_VALUES);

    cout << "U+_01=" << -PETSC_i*sintp*sin(PETSC_PI-phi_plus) << endl;
   MatSetValue(_CTX->U_minus_gates[r], i, j, -PETSC_i*sintm*sin(PETSC_PI-phi_minus), INSERT_VALUES);
   cout << "U-_01=" << -PETSC_i*sintm*sin(PETSC_PI-phi_minus) << endl;
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
    b.flip(r2);
    int j = (int)(b.to_ulong());
    if ((j >= _Iend) or (j < _Istart)) { o_nnz[i - _Istart]++; }
      else { d_nnz[i - _Istart]++;}
  }

  return std::make_tuple(d_nnz, o_nnz);
}

#endif





