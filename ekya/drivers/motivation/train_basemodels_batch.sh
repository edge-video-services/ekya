#!/bin/bash
set -e

# Num hidden
for model_str in resnet18 resnet50 resnet101; do
  # Num hidden
  for nh in 64 512 1024; do
    echo Training $nh $model_str
    python train_basemodel.py -e 100 -dc -mn ${model_str} -nh ${nh} -r /home/researcher/datasets/cityscapes/ -cp /home/researcher/models/pretrained_cityscapes_fftmunster_${model_str}_${nh}x2.pt --lists-train frankfurt,munster --lists-val lindau,tubingen
  done
done
