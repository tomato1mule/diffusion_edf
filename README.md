# Diffusion EDF
Official implementation of the paper 'Diffusion-EDFs: Bi-equivariant Denoising Generative Modeling on SE(3) for Visual Robotic Manipulation'

Paper: https://arxiv.org/abs/2309.02685
# Installation

**Step 1.** Clone Github repository.
```shell
git clone --recurse-submodules https://github.com/tomato1mule/diffusion_edf
```

**Step 2.** Setup Conda/Mamba environment. We recommend using Mamba for faster installation.
```shell
conda install mamba -c conda-forge
mamba create -n diff_edf python=3.8
conda activate diff_edf
```

**Step 3.** Install Diffusion EDF.
```shell
pip install torch==1.13.1 torchvision==0.14.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -e .
```
**Step 3.** Install EDF Interface.
```shell
cd edf_interface
pip install -e . # If error occurs, please check in step 1 you have correctly cloned with '--recurse-submodules' flag.
```

# Usage
## Training
```shell
bash train_pick.bash
bash train_place.bash
```
To see running experiments, use tensorboard:
```shell
tensorboard --logdir=./runs
```
## Evaluation
Please open `evaluate.ipynb` with Jupyter notebook.

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
> Demonstration files are saved in meter units. Therefore, rescaling is defined in the `train_configs.yaml`.
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

# Citation
Please consider citing our paper if you find it helpful.

```
@article{ryu2023diffusion,
  title={Diffusion-EDFs: Bi-equivariant Denoising Generative Modeling on SE (3) for Visual Robotic Manipulation},
  author={Ryu, Hyunwoo and Kim, Jiwoo and Chang, Junwoo and Ahn, Hyun Seok and Seo, Joohwan and Kim, Taehan and Choi, Jongeun and Horowitz, Roberto},
  journal={arXiv preprint arXiv:2309.02685},
  year={2023}
}
```

