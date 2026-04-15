#!/bin/bash
#SBATCH --job-name=stwos_eval_strong
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:V100:1
#SBATCH -t 02:00:00
#SBATCH --output=stwos_eval_strong-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tkhandwekar3@gatech.edu

module load anaconda3
conda activate ~/scratch/soccertwos_env

cd /home/hice1/tkhandwekar3/soccer-twos-starter

# If you did not package a checkpoint yet, point directly to one:
# export STRONG_AGENT_CHECKPOINT=/home/hice1/tkhandwekar3/scratch/ray_results/PPO_STRONG_SELFPLAY_SHAPED/<trial>/checkpoint_XXXX/checkpoint-XXXX

/home/hice1/tkhandwekar3/scratch/soccertwos_env/bin/python evaluate_vs_baseline.py \
  --agent1 my_strong_agent \
  --agent2 ceia_baseline_agent \
  --episodes 200
