#!/bin/bash
set -e

for ALLOC in 10 20 30 40 50 60 70 80 90 100; do
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${ALLOC} python driver_profiles.py \
  -dc \
  -r /home/researcher/datasets/cityscapes/ \
  -rp /home/researcher/models/pretrained_cityscapes_fftmunster_ \
  --results-path /tmp/benchmarking_results_bs8/ \
  -nt 10 \
  -nsp 0 \
  -dtfs \
  --validation-frequency 8 \
  --lists-train zurich \
  --epochs 8 \
  --override-batch-size 8
done
