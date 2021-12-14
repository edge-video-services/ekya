#!/bin/bash
# a script to setup the environment for nvidia mps. Run this as sudo!
set -e

## Enable exclusive mode
# Stop any display processes
service gdm stop

# Set exclusive mode
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# Reboot GPU
nvidia-smi -i 0 -r

# Start MPS daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
mkdir /tmp/nvidia-log
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d