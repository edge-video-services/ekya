#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='../../dataset/waymo/waymo_classification_images'
MODEL_PATH='../../pretrained_models/waymo'
HYPS_PATH='../experiment_drivers/profiling/hyp_map_only18.json'
INFERENCE_PROFILE_PATH='../experiment_drivers/real_inference_profiles.csv'

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
DATASET_NAME=waymo
NUM_CLASSES=4
EXP_NAME=model_cache_exp
OUTPUT_ROOT=../../results
GOLDEN_MODEL_PATH=../../golden_model/coco_resnext101_elastic.pth.tar

# Reduce number of epochs when using past retraining data
for CITY in sf_000_009 sf_020_029 sf_030_039 sf_050_059 sf_060_069 sf_070_079 sf_080_089; do
  for HPARAM_ID in 0 1 2 3 4 5; do
    echo Running profiling on city $CITY Hparam: $HPARAM_ID
    CUDA_VISIBLE_DEVICES=2 python ../experiment_drivers/driver_multicity.py \
             --scheduler profiling \
             --cities ${CITY} \
             --log-dir ${OUTPUT_ROOT}/${EXP_NAME}/golden_label_logs \
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
             --gpu-memory 4 \
             --profile-write-path ${OUTPUT_ROOT}/${EXP_NAME}/golden_label_profiles

  done
done
