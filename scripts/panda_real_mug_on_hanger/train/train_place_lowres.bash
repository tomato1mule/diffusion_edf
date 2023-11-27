echo "Train Low-resolution Score-matching Model"

configs_root_dir="configs/panda_mug/place_lowres"
train_configs_file="train_configs.yaml"
task_configs_file="task_configs.yaml"

PYTHONHASHSEED=0 python3 diffusion_edf/train.py --configs-root-dir=$configs_root_dir \
                                                --train-configs-file=$train_configs_file \
                                                --task-configs-file=$task_configs_file \
                                                --log-name-postfix="Place_LowRes_Panda_Mug_on_Hanger"