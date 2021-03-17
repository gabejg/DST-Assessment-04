#!/bin/bash -login

#SBATCH --job-name=NNtest
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=100000M

echo "loading Anaconda and TensorFlow"
module add languages/anaconda3/2020-3.8.5
module add languages/anaconda3/2020.02-tflow-2.2.0 

echo "moving to submit directory $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
python Assessment_4_Matt.py &> Assessment_4_Matt.log