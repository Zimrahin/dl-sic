#!/bin/bash

#SBATCH --job-name=train_tdcr
#SBATCH --cpus-per-gpu=1         # Dataloader num_workers + 1 (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:1             # Number and type of GPU cards and type allocated
#SBATCH --mem=16G                # Total memory allocated
#SBATCH --time=10:00:00          # Total run time limit (HH:MM:SS)
#SBATCH --output=%x_%j.out       # Output file name
#SBATCH --exclude=gpu009,gpu01[2-3],gpu01[5-7]  # Exclude team nodes

echo "### Running $SLURM_JOB_NAME ###"

set +x  # Turn off verbose mode
cd ${SLURM_SUBMIT_DIR}

module purge

# Set Conda environment
source /home/$USER/.bashrc
# PyTorch environment should be created previously
source activate pt_env

python -u train.py \
    --model_type "complex" \
    --dtype "complex64" \
    --batch_size 1 \
    --epochs 50 \
    --learning_rate 1e-3 \
    --weight_decay 0 \
    --val_split 0.2 \
    --target 1 \
    --num_workers 0 \
    --model_param_M 128 \
    --model_param_N 32 \
    --model_param_U 128 \
    --model_param_V 8 \
    --checkpoints_dir "./checkpoints" \
    --dataset_path "./data/simulated_dataset.pt" \
    --runtime_gen
