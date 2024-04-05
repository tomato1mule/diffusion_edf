# [CVPR 2024 Highlight] Diffusion-EDFs
![plot](https://github.com/tomato1mule/diffusion_edf/blob/main/figures/panda_pick.gif?raw=true)
![plot](https://github.com/tomato1mule/diffusion_edf/blob/main/figures/panda_place.gif?raw=true)

Official implementation of the paper 'Diffusion-EDFs: Bi-equivariant Denoising Generative Modeling on SE(3) for Visual Robotic Manipulation' (CVPR 2024, Highlight)

**Project Website:** https://sites.google.com/view/diffusion-edfs

**Paper:** https://arxiv.org/abs/2309.02685
# Installation

**Step 1.** Clone Github repository.
```shell
git clone --recurse-submodules https://github.com/tomato1mule/diffusion_edf
```
> [!IMPORTANT]
> You must RECURSIVELY clone the repositories. Please also use github LFS to clone ```demo``` and ```checkpoints``` directories. Without LFS, they would appear empty.


**Step 2.** Setup Conda/Mamba environment. We recommend using Mamba for faster installation.
```shell
conda install mamba -c conda-forge
mamba create -n diff_edf python=3.8
conda activate diff_edf
```

**Step 3.** Install Diffusion EDF.
```shell
mamba install -c conda-forge cxx-compiler==1.5.0
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --no-index torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -e .
```

**Step 4.** Install EDF Interface.
```shell
cd edf_interface
pip install -e . # If error occurs, please check in step 1 you have correctly cloned with '--recurse-submodules' flag.
cd ..
```

# Quickstart
## Evaluation
Open the ```evaluate_<task_name>.ipynb``` file using jupyter notebook to see how Diffusion-EDFs work.
> [!TIP]
> We provide three real-world manipulation examples with Franka Panda robot.
> * ```evaluate_real_mug.ipynb```
> * ```evaluate_real_bowl.ipynb```
> * ```evaluate_real_bottle.ipynb```



## Training
```shell
bash scritps/<task_name>/train.bash
```
> [!TIP]
> We provide three real-world manipulation examples with Franka Panda robot.
> * ```bash scripts/panda_real_mug_on_hanger/train.bash```
> * ```bash scripts/panda_real_bowl_on_dish/train.bash```
> * ```bash scripts/panda_real_bottle_on_shelf/train.bash```

To see logs for running experiments, use tensorboard:
```shell
tensorboard --logdir=./runs
```


# Doc
**Inputs**

* **scene_input, grasp_input**: FeaturedPoints (NamedTuple)
    - FeaturedPoints.**x**: 3d position of the points; Shape: (nPoints, 3)
    - FeaturedPoints.**f**: Feature vector of the points; Shape: (nPoints, dim_feature)
    - FeaturedPoints.**b**: Minibatch index of each points. Currently all set to zero; Shape:(nPoints,)
    - FaturedPoints.**w**: Optional point attention value; Shape: (nPoints, )
* **T_seed**: Initial pose to start denoising process; Shape: (nPoses, 7)
    - **T_seed[..., :4]**: Quaternion (qw, qx, qy, qz)
    - **T_seed[..., 4:]**: Position (x, y, z)

> [!IMPORTANT]
> Properly setting the unit system for position is crucial. In this code, centimeter unit is used for the model. For example, the distance between two points (x=0., y=0., z=0.) and (x=1., y=0., z=0.) is 1cm.

> [!NOTE]
> Demonstration files are saved in meter units. Therefore, rescaling is defined in the `train_configs.yaml`.
> For example, 
> #### **`configs/panda_mug/pick_lowres/train_configs.yaml`**
> ```yaml
> rescale_factor: &rescale_factor 100.0 # Meters to Centimeters
> preprocess_config:
>   - name: "Downsample"
>     kwargs:
>       voxel_size: 0.01 # In Meters
>       coord_reduction: "average"
>   # ...
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

