#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
##########SBATCH --time=4:59:00
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --job-name swirl_specs
#SBATCH --output=swirl_spec_%j.txt
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=10
export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp

module load NiaEnv/2019b intelpython3/2019u5 gnu-parallel
source activate $HOME/../macdo708/.conda/envs/aneurisk_librosa
cd $SLURM_SUBMIT_DIR

mkdir swirl_files/case_A/specs && mkdir swirl_files/case_A/imgs 

cd $PWD/swirl_files/case_A/case_028_low 
python ../../../compute_swirl_spectrogram.py . swirl_028_low  ../specs ../imgs 1