#!/bin/bash
#SBATCH --job-name=stwos_strong_train
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:V100:1
#SBATCH -t 18:00:00
#SBATCH --output=stwos_strong_train-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tkhandwekar3@gatech.edu

module load anaconda3
conda activate ~/scratch/soccertwos_env

cd /home/hice1/tkhandwekar3/soccer-twos-starter

ray stop --force
mkdir -p ~/scratch/ray_results
mkdir -p ~/scratch/ray_tmp
export RAY_TMPDIR=~/scratch/ray_tmp

/home/hice1/tkhandwekar3/scratch/soccertwos_env/bin/python train_strong_selfplay_shaped.py
