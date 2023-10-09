#!/bin/bash

echo "Train Low-resolution Score-matching Model"

configs_root_dir="configs/panda_mug/place_lowres"
train_configs_file="train_configs.yaml"
task_configs_file="task_configs.yaml"

PYTHONHASHSEED=0 python3 diffusion_edf/train.py --configs-root-dir=$configs_root_dir \
                                                --train-configs-file=$train_configs_file \
                                                --task-configs-file=$task_configs_file \
                                                --log-name-postfix="Place_LowRes_Panda"

echo "Train Super-resolution Score-matching Model"

configs_root_dir="configs/panda_mug/place_highres"
train_configs_file="train_configs.yaml"
task_configs_file="task_configs.yaml"

PYTHONHASHSEED=0 python3 diffusion_edf/train.py --configs-root-dir=$configs_root_dir \
                                                --train-configs-file=$train_configs_file \
                                                --task-configs-file=$task_configs_file \
                                                --log-name-postfix="Place_HiRes_Panda"

echo "Train EBM Model"

configs_root_dir="configs/panda_mug/place_ebm"
train_configs_file="train_configs.yaml"
task_configs_file="task_configs.yaml"

PYTHONHASHSEED=0 python3 diffusion_edf/train.py --configs-root-dir=$configs_root_dir \
                                                --train-configs-file=$train_configs_file \
                                                --task-configs-file=$task_configs_file \
                                                --log-name-postfix="Place_EBM_Panda"
                                                