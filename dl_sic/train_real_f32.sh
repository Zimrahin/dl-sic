#!/bin/bash

python train.py \
    --model_type "real" \
    --dtype "float32" \
    --batch_size 1 \
    --epochs 50 \
    --learning_rate 1e-3 \
    --weight_decay 0 \
    --val_split 0.2 \
    --target 1 \
    --dataset_path "../data/simulated_dataset.pt" \
    --num_workers 0 \
    --model_param_M 128 \
    --model_param_N 64 \
    --model_param_U 128 \
    --model_param_V 8 \
    --checkpoints_dir "./checkpoints/real_f32"
