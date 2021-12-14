#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_NAME=waymo
DATASET_PATH=../../datasets/waymo/waymo_classification_images
MODEL_PATH=../../pretrained_models/waymo
GOLDEN_MODEL_PATH=../../golden_model/coco_resnext101_elastic.pth.tar
RESULTS_ROOT=../../results
HYPS_PATH='./profiling/hyp_map_only18.json'
INFERENCE_PROFILE_PATH='real_inference_profiles.csv'

# Large number to give enough time to complete. Tasks terminate early when
# retraining finishes so this is not a problem.
RETRAINING_PERIOD=9999999
NUM_TASKS=10
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=30
START_TASK=1
TERMINATION_TASK=9
MAX_INFERENCE_RESOURCES=0.25
NUM_CLASSES=4

# Reduce number of epochs when using past retraining data
# for CITY in phx_20_29 phx_40_49; do

for CITY in phx_020_029 phx_040_049 sf_060_069 sf_020_029 sf_000_009 sf_050_059 sf_070_079 sf_080_089 phx_050_059 phx_030_039 sf_030_039; do #phx_30_39 phx_40_49 phx_50_59; do
  for HPARAM_ID in 0 1 2 3 4 5; do
    echo Running profiling on city $CITY Hparam: $HPARAM_ID
    CUDA_VISIBLE_DEVICES=0 python driver_multicity.py \
             --scheduler profiling \
             --cities ${CITY} \
             --log-dir ${RESULTS_ROOT}/${DATASET_NAME}_outputs/golden_label_logs \
             --retraining-period ${RETRAINING_PERIOD} \
             --num-tasks ${NUM_TASKS} \
             --inference-chunks ${INFERENCE_CHUNKS} \
             --num-gpus ${NUM_GPUS} \
             --num-classes ${NUM_CLASSES} \
             --dataset-name ${DATASET_NAME} \
             --root ${DATASET_PATH} \
             --use-data-cache \
             --restore-path ${MODEL_PATH} \
             --lists-pretrained pretrain_2000 \
             --hyperparameter-id $HPARAM_ID \
             --start-task ${START_TASK} \
             --termination-task ${TERMINATION_TASK} \
             --epochs ${EPOCHS} \
             --hyps-path ${HYPS_PATH} \
             --inference-profile-path ${INFERENCE_PROFILE_PATH} \
             --max-inference-resources ${MAX_INFERENCE_RESOURCES} \
             --profiling-mode \
             --golden-label \
             --golden-model-ckpt-path ${GOLDEN_MODEL_PATH} \
             --train-split 0.7 \
             --profile-write-path ${RESULTS_ROOT}/${DATASET_NAME}_outputs/golden_label_profiles\
             --gpu-memory 4
  done
done
