#!/bin/bash -login

#SBATCH --job-name=MyNN
#SBATCH --partition=cpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=100000M

echo "loading R"
module add languages/r/4.0.2

echo "moving and submit directory"
cd $SLURM_SUBMIT_DIR
Rscript NN.r &> MyNN.log

