#!/bin/bash
# This file contains bash commands that will be executed at the beginning of the container build process,
# before any system packages or programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it
<<<<<<< HEAD
=======
# Install system dependencies first
sudo apt-get update && apt-get install -y \
    git python3-dev python3-pip build-essential \
    libcpuinfo-dev \
    libgloo-dev \
    libsleef-dev \
    libpthread-stubs0-dev \
    libeigen3-dev \
    libprotobuf-dev \
    protobuf-compiler

# Clone and build PyTorch with system libraries
sudo git clone --recursive https://github.com/pytorch/pytorch.git \
    && cd pytorch \
    && git checkout v2.4.0 \
    && pip install setuptools wheel \
    && TORCH_CUDA_ARCH_LIST="7.0+PTX" \
    CUDA_HOME=/usr/local/cuda \
    CMAKE_ARGS="-DUSE_SYSTEM_LIBS=ON" \
    python3 setup.py install
>>>>>>> 7c227c6 (Changed env w/ apt & pip & more...)
