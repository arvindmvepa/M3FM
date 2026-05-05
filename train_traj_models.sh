#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --model_type autoreg_mlp
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --model_type residual_mlp
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --model_type transformer
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --model_type gru
CUDA_VISIBLE_DEVICES=$1 python train_embedding_trajectory.py --model_type lowrank_linear
