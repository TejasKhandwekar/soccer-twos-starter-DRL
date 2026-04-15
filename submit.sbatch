#!/bin/bash
#SBATCH --job-name=soccerstwos
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G 
#SBATCH --gres=gpu:V100:1
#SBATCH -t 12:00:00
#SBATCH --output=soccerstwos-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tkhandwekar3@gatech.edu

module load anaconda3
conda activate ~/scratch/soccertwos_env

# CLEANUP: Kill any hanging ray processes and clear the temp folder
ray stop --force
rm -rf ~/scratch/ray_tmp/*

python train_ray_ppo_ma_teams.py