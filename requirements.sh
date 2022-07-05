#!/bin/bash

ENV_NAME="iba_inf"
CUDA_VERSION="10.2"

conda create --name $ENV_NAME python=3.6 -y
conda install -n $ENV_NAME pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=$CUDA_VERSION -c pytorch -y
conda install -n $ENV_NAME numpy scipy matplotlib scikit-learn scikit-image pandas opencv tqdm -y
conda install -n $ENV_NAME -c https://conda.anaconda.org/simpleitk SimpleITK -y
conda install -n $ENV_NAME -c conda-forge nibabel -y

