#!/bin/bash -login

#SBATCH --job-name=BinaryNN
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000M

echo "loading R"
module add languages/r/4.0.2

echo "Running the script"
cd $SLURM_SUBMIT_DIR
Rscript NNPackage.r &> MySecondaryNN.log

