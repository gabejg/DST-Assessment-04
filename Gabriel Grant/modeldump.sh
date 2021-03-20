#!/bin/bash -login
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=nohidettf
#SBATCH --mem=16000M

echo "Loading TensorFlow"
module add languages/anaconda2/5.3.1.tensorflow-1.12
module add languages/anaconda3/2020.02-tflow-2.2.0

echo "moving to submit directory $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
./gpujob

python3 nohide.py &> nohide.log
