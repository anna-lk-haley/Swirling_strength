#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --job-name PSD
#SBATCH --output=PSD_%j.txt
#SBATCH --mail-type=FAIL

#export OMP_NUM_THREADS=13
export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp

module load CCEnv StdEnv/2020 gcc/9.3.0 python/3.9.6 #petsc/3.20.0
source $HOME/.virtualenvs/pv_updated/bin/activate

python $PROJECT/Swirl/scripts/make_PSD.py PerturbNewt500 swirl_files/fig_data/PerturbNewt500/PerturbNewt500_swirl_100_nsubs100_ndivs100.npz swirl_files/figs/PerturbNewt500
python $PROJECT/Swirl/scripts/make_PSD.py PerturbNewt400 swirl_files/fig_data/PerturbNewt400/PerturbNewt400_swirl_100_nsubs100_ndivs100.npz swirl_files/figs/PerturbNewt400
python $PROJECT/Swirl/scripts/make_PSD.py PerturbNewt600 swirl_files/fig_data/PerturbNewt600/PerturbNewt600_swirl_100_nsubs100_ndivs100.npz swirl_files/figs/PerturbNewt600