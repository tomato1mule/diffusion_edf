#!/bin/bash

scripts_root_dir=scripts/panda_real_bowl_on_dish/train/



# source  ~/.bashrc
# conda activate diff_edf

source $scripts_root_dir/train_pick_lowres.bash
source $scripts_root_dir/train_pick_highres.bash
source $scripts_root_dir/train_pick_ebm.bash
source $scripts_root_dir/train_place_lowres.bash
source $scripts_root_dir/train_place_highres.bash
source $scripts_root_dir/train_place_ebm.bash