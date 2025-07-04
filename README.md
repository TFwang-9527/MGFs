# MGFs: Masked Gaussian Fields for Meshing Building based on Multi-View Images
[Tengfei Wang](https://github.com/TFwang-9527), Zongqian Zhan*, Rui Xia, Linxia Ji, Xin Wang*.
### [Project Page](https://tfwang-9527.github.io/MGFs/) | [arXiv](https://arxiv.org/abs/2408.03060)

# Installation
Clone the repository and create an anaconda environment using
```
git clone git@github.com:autonomousvision/gaussian-opacity-fields.git
cd gaussian-opacity-fields

conda create -y -n gof python=3.8
conda activate gof

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

# tetra-nerf for triangulation
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
# you can specify your own cuda path
# export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
make 
pip install -e .
```

# Dataset
The cropped Barn.ply is [here](https://drive.google.com/file/d/1r924zwKXOKB2IyFV04WHBTzqqTBdtf_r/view?usp=drive_link])
 ![image](https://github.com/user-attachments/assets/7dcd8cc4-3875-46ec-9a0d-d485f9624b19)

# training
First, obtain the building mask, multi_mask, and white images, then place these three folders in the same directory as the image folder.

python train.py -s TNT_GOF/TrainingSet/Barn -m exp_TNT/Barn -r 2 --use_decoupled_appearance

# extract the mesh after training
python extract_mesh.py -m exp_TNT/Barn --iteration 30000
```

# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [GOF]([https://github.com/autonomousvision/mip-splatting](https://github.com/autonomousvision/gaussian-opacity-fields)).[2DGS]([https://github.com/hbb1/2d-gaussian-splatting])

# Citation
