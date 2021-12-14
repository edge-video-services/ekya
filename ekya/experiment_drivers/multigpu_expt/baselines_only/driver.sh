#!/bin/bash
set -e

# CUDA_VISIBLE_DEVICES=0,1,2,3 ./4gpu.sh
CUDA_VISIBLE_DEVICES=0,1 ./2gpu.sh
CUDA_VISIBLE_DEVICES=0 ./1gpu.sh

