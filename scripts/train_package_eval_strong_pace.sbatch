#!/bin/bash
#SBATCH --job-name=stwos_train_eval_auto
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:V100:1
#SBATCH -t 16:00:00
#SBATCH --output=stwos_train_eval_auto-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tkhandwekar3@gatech.edu

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/soccertwos_env

cd /home/hice1/tkhandwekar3/soccer-twos-starter

# Keep temporary files and logs in scratch.
mkdir -p ~/scratch/ray_results
mkdir -p ~/scratch/ray_tmp
mkdir -p ~/scratch/soccer_twos_reports
export RAY_TMPDIR=~/scratch/ray_tmp

# Configure a full run that fits the PACE GPU walltime limit.
export STRONG_TRAIN_HOURS=16
export STRONG_TRAIN_TIMESTEPS=40000000
export STRONG_BASE_PORT=$((15000 + SLURM_JOB_ID % 40000))
echo "Using STRONG_BASE_PORT=${STRONG_BASE_PORT}"

START_TS=$(date +%Y%m%d_%H%M%S)
REPORT_DIR=~/scratch/soccer_twos_reports/${START_TS}
mkdir -p "${REPORT_DIR}"

echo "[1/4] Training strong agent..."
ray stop --force || true
/home/hice1/tkhandwekar3/scratch/soccertwos_env/bin/python train_strong_selfplay_shaped.py | tee "${REPORT_DIR}/train.log"

echo "Training will finish after 16 hours. Package/eval can be run afterward with the checkpoint from ray_results."
