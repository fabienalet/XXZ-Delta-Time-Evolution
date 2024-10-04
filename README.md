# XXZ-Delta-Time-Evolution

Time evolution after a quench using Krylov methods.


TODO : 
No documentation yet. More details later.


# Step 0a : Install Petsc on some cluster

How to install release (example on lptsv7):

In the following `path-to-home` is perhaps something like `/home/ahaldar` 

```
cd ~; mkdir src; cd src
wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.21.5.tar.gz; tar xvfz petsc-3.21.5.tar.gz
```
Create a configuration file ``configure_petsc.sh`` as this:
```
cd petsc-3.21.5
module purge

# Load the available modules on the wanted cluster. Those are for lptsv7
module load intel
module load impi
source /opt/intel/oneapi/mpi/2021.5.1/env/vars.sh
module load cmake

# Here we configure for complex numbers (ARCH is called complex)
./configure --force --COPTFLAGS="-O3" --CXXOPTFLAGS="-O3" --FOPTFLAGS="-O3" --with-blas-lapack-dir=$MKLROOT --with-debugging=0 --with-errorchecking=0 --with-openmp --with-precision=double --with-scalar-type=complex --with-make-np=1 MPI-DIR=$MPI_DIR PETSC_ARCH=complex
```
then ```bash configure_petsc.sh``` then configuration tells you which command line you would like to run, it will be something like 

```make PETSC_DIR=path-to-home/src/petsc-3.21.5 PETSC_ARCH=complex all```

You can want to save this instruction in a script called ```make_petsc.sh``` for future use
```
cd petsc-3.21.5
module purge
module load intel
module load impi
module load cmake
export CC=mpiicc
export CXX=mpiicpc
make PETSC_DIR=path-to-home/src/petsc-3.21.5 PETSC_ARCH=complex all
```

After you run this command `bash make_petsc.sh``, Petsc should be correctly installed

# Step 0b : Install Slepc 

```
cd path-to-home/src
wget https://slepc.upv.es/download/distrib/slepc-3.21.2.tar.gz; tar xvfz slepc-3.21.2.tar.gz
```
Create a configuration file script.conf
```
module load intel
module load impi
module load cmake
export CC=mpiicc
export CXX=mpiicpc
export PETSC_DIR=path-to-home/src/petsc-3.21.5
export PETSC_ARCH=complex
export SLEPC_DIR=path-to-home/src/slepc-3.21.2
./configure
```
then ```bash script.conf``` then configuration tells you which command line to run, something like that ```make SLEPC_DIR=/tmpdir/alet/newsrc/slepc-3.21.2 PETSC_DIR=/tmpdir/alet/newsrc/petsc-3.21.5 PETSC_ARCH=real```

# Step 1 : Install this git directory

``` cd ~/src; git clone git@github.com:fabienalet/XXZ-Delta-Time-Evolution.git ```


## For olympe

```cd ~;mkdir build; mkdir build/XXZ-Delta-Time-Evolution; cd build/XXZ-Delta-Time-Evolution```
Create a ```script.build``` file such as (CAREFUL : Change the correct path-to-gome Petsc and Slepc directory below !!):
```
module purge
module load intel
module load intelmpi
module load cmake
module load gcc/7.3.0
export CC=icc
export CXX=icpc
export PETSC_DIR=path-to-home/src/petsc-3.20.5
export SLEPC_DIR=path-to-home/src/slepc-3.20.4
export PETSC_ARCH=complex
export ED_PETSC_ARCH=$PETSC_ARCH
cmake path-to-home/src/XXZ-Delta-Time-Evolution
```

then ```bash script.build```, then ```make xxz_krylov```
(If the make command fail, re-load the modules by doing ``module purge; module load intel;module load intelmpi;module load cmake;module load gcc/7.3.0``


## For lptsv7
This may require that you create a public ssh-key on lptsv7, and then add it to your Github account (this can be a bit troublesome, but the info below is useful
(https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
Do the steps Generating a new SSH key, Adding your SSH key to the ssh-agent (I think the `UseKeychain yes` line should be skipped if you don't use a passphrase), and that should work.

When the code is well pulled, continue with

```cd ~;mkdir build; mkdir build/XXZ-Delta-Time-Evolution; cd build/XXZ-Delta-Time-Evolution```
Create a ```script.build``` file such as :
```
module purge
module load intel
module load impi
module load cmake
export CC=mpiicc
export CXX=mpiicpc
#export PETSC_DIR=path-to-home/src/petsc/
export PETSC_DIR=path-to-home/src/petsc-3.21.5
export PETSC_ARCH=complex
export ED_PETSC_ARCH=$PETSC_ARCH
#cmake path-to-home/src/XXZ-Delta-Time-Evolution
cmake /home/prowal/src/XXZ-Delta-Time-Evolution
```

then ```bash script.build```, then ```make xxz_krylov```
(If the make command fail, re-load the modules by doing ``module purge; module load intel;module load impi;module load cmake``

# Running the code : Direct Usage
```mkdir test;cd test```
Create a ```krylov.options``` file such as :
```
-L 16
-disorder 5.0
-seed 3
-tmax 100
-pbc 0

#-measure_imbalance
#-measure_entanglement
```
Usage
```../xxz_krylov -measure_entanglement 1 -meaure_imbalance 1```
```mpirun -np 10 ../xxz_krylov_te```

# Running the code : Slurm Script 
Not correct at the moment

```
#!/bin/bash
#SBATCH -J 
#SBATCH -N 2
#SBATCH -n 48
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --time 03:00:00


module purge
module load intel
module load impi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
## Make sure there is a floquet.options file in this test repertory that is adapted
## command line options take precedence over options in the floquet.options file
/usr/bin/time -v srun  ../xxz_krylov_te -L 16 -seed 451  > test.out.txt 2> test.err.txt
```

Alternative : to be tested
```
srun -N nodes --ntasks=ntasks --ntasks-per-node=n -c ncpu-per-task ./xxz_krylov_te
```


# LPTSV7 specific subtleties

At the moment when loading the module intel, one obtains
```
$ldd xxz_krylov_te
...
libquadmath.so.0 => /usr/lib64/libquadmath.so.0 (0x00007f7fce50c000)
...
```
on the login node, and
```
...
libquadmath.so.0 => not found
...
```
on the compute nodes.
As a temporary fix one can copy this library to the build folder and add the local path to the `LD_LIBRARY_PATH`
```
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
```



