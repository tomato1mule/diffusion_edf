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
# <WIP> bash train_place.bash
```
To see running experiments, use tensorboard:
```shell
tensorboard --logdir=./runs
```
## Evaluation
Please open *'evaluate_pick.ipynb'* with Jupyter notebook.

*'evaluate_place.ipynb'* is work in progress.

**Inputs**

* **scene_input, grasp_input**: FeaturedPoints (NamedTuple)
    - FeaturedPoints.**x**: 3d position of the points; Shape: (nPoints, 3)
    - FeaturedPoints.**f**: Feature vector of the points; Shape: (nPoints, dim_feature)
    - FeaturedPoints.**b**: Minibatch index of each points. Currently all set to zero; Shape:(nPoints,)
    - FaturedPoints.**w**: Optional point attention value; Shape: (nPoints, )
* **T_seed**: Initial pose to start denoising process; Shape: (nPoses, 7)
    - **T_seed[..., :4]**: Quaternion (qw, qx, qy, qz)
    - **T_seed[..., 4:]**: Position (x, y, z)

> **Note**\
> Properly setting the unit system for position is crucial. In this code, centimeter unit is used for the model. For example, the distance between two points (x=0., y=0., z=0.) and (x=1., y=0., z=0.) is 1cm.

> **Warning**\
> Demonstration files are saved in meter units. Therefore, rescaling is defined in the 'train_configs.yaml'.
> #### **`configs/pick_lowres/train_configs.yaml`**
> ```yaml
> rescale_factor: &rescale_factor 100.0 # Meters to Centimeters
> preprocess_config:
>   - name: "Downsample"
>     kwargs:
>       voxel_size: 0.01 # In Meters
>       coord_reduction: "average"
>   - name: "Rescale"
>     kwargs:
>       rescale_factor: *rescale_factor
>```
> Note that the voxel size of the voxel downsample filter in the above config file is 0.01m = 1cm. Note that 'train_configs.yaml' and 'task_configs.yaml' use meter units while 'score_model_configs' use centimeter units.


