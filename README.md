## SC-VI-SOD
This project is used for paper (SCVI: A Semi-Coupled Visible-Infrared Small Object Detection Method based on Multimodal Proposal-level Probability Fusion Strategy). It is modified from mmdetection.

## Enviroment
CUDA 12.1
Torch 2.4.1
We use the docker to run this code. The docker image is pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel.

## Install
# Conda
conda create -n scvi python=3.10 -y
(Optional 1) conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
(Optional 2) pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4, <2.2.0"
pip install fairscale, einops, pycocotools==2.0.8
pip install -v -e .
# Docker
(sudo) docker run --gpus all -itd --rm -p 8888:8888 --name scvi -v ~:/workspace --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel bash
apt-get install ffmpeg libsm6 libxext6  -y
apt-get update && apt-get install libgl1
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4, <2.2.0"
pip install -v -e .
pip install opencv-python-headless
pip install scikit-learn==1.6.0

## Run
export CUDA_VISIBLE_DEVICES=0,1,2,3; tools/dist_train.sh configs/mine/parallel/dino-parallel-0716_r50_8xb2_1x_rgbt_tiny.py 4 --work-dir results/
