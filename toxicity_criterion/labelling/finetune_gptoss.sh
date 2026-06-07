#!/bin/bash
# SLURM job for the GPT-OSS 20B LoRA fine-tune (stratifier; document-only). Adjust the
# SBATCH directives, module versions, and paths to your cluster. Set PROJECT_DIR to wherever
# you checked out this repository.
#SBATCH --job-name=gptoss_tox_train
#SBATCH --output=gptoss_tox_train_%j.log
#SBATCH --error=gptoss_tox_train_%j.err
#SBATCH --time=09:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --partition=main-gpu
#SBATCH --account=[your-slurm-account]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=[your-email]

source /etc/profile.d/modules.sh
module load nvidia/cuda-11.8

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"

nvidia-smi

python -m toxicity_criterion.labelling.finetune_gptoss

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Date: $(date)"
