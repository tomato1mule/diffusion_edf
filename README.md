# Diffusion EDF
Blah Blah
# Installation

**Step 1.** Clone Github repository.
```shell
git clone https://github.com/tomato1mule/diffusion_edf
```

**Step 2.** Setup Conda/Mamba environment. We recommend using Mamba for faster installation.
```shell
# if you don't have mamba yet, install it first (not needed when using mambaforge):
conda install mamba -c conda-forge

# now create a new environment
mamba create -n diff_edf python=3.8
conda activate diff_edf
```

**Step 3.** Install Diffusion EDF.
```shell
pip install torch==1.13.1 torchvision==0.14.1
pip install theseus-ai==0.1.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
# mamba install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# mamba install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
#mamba install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
# mamba install -c fvcore -c iopath -c conda-forge fvcore==0.1.5.post20221221 iopath==0.1.9
# mamba install pytorch3d==0.7.2 -c pytorch3d
# mamba install pyg==2.2.0 -c pyg
# mamba install pytorch-sparse==0.6.15 pytorch-scatter==2.0.9 pytorch-cluster==1.6.0 -c pyg
# mamba install pytorch-sparse==0.6.17 pytorch-scatter==2.1.1 pytorch-cluster==1.6.1 -c pyg
pip install -e .
```

# Usage
## Training
```shell
bash train_pick.bash
<WIP> bash train_place.bash
```
To see running experiments, use tensorboard:
```shell
tensorboard --logdir=./runs
```
## Evaluation
Please open *'evaluate_pick.ipynb'*$\ $ with Jupyter notebook.

*'evaluate_pick.ipynb'*$\ $ is work in progress.