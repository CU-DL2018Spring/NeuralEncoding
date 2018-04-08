#!/bin/bash

# Install CUDA 9.0
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo apt-get update
sudo apt-get install cuda=9.0.176-1
sudo apt-get install libcudnn7-dev
sudo apt-get install libnccl-dev

# Install virtual environment
conda create -n tensorflow pip python=3.6
source activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl

conda install -c conda-forge keras
pip install tableprint
conda install -c conda-forge deepdish
pip install pyret


# Download dataset
sudo apt-get install unzip
mkdir -p /home/ubuntu/experiments/data/
cd /home/ubuntu/experiments/data/
wget https://www.dropbox.com/s/pslelorr1xtnlko/15_10_07_retina_dataset.zip?dl=0 15_10_07_retina_dataset.zip
unzip 15_10_07_retina_dataset.zip
mv 15_10_07_retina_dataset 15-10-17
