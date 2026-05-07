#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --base_loss huber --model_type autoreg_mlp --aux_mode joint --aux_style attached --aux_loss_weight 0.1
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --base_loss huber --model_type residual_mlp --aux_mode joint --aux_style attached --aux_loss_weight 0.1
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --base_loss huber --model_type transformer --aux_mode joint --aux_style attached --aux_loss_weight 0.1
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --base_loss huber --model_type gru --aux_mode joint --aux_style attached --aux_loss_weight 0.1
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --base_loss huber --model_type lowrank_linear --aux_mode joint --aux_style attached --aux_loss_weight 0.1