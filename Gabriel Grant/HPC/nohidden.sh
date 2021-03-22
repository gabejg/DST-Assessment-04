#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=nohiddengpujob

echo "loading TensorFlow"
module add languages/anaconda3/3.5-4.2.0-tflow-1.7

echo "moving to submit directory $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
./gpujob

python nohidden.py &> nohidden.log
