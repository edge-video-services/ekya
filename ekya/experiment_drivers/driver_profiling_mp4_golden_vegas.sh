#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure
# retraining results in accuracy gains.
set -e

DATASET_NAME=vegas
DATASET_PATH=../../dataset/${DATASET_NAME}
MODEL_PATH=../../pretrained_models/${DATASET_NAME}
GOLDEN_MODEL_PATH=../../golden_model/coco_resnext101_elastic.pth.tar
RESULTS_ROOT=../../results
HYPS_PATH='profiling/hyp_map_all.json'
INFERENCE_PROFILE_PATH='real_inference_profiles.csv'
RETRAINING_PERIOD=9999999 # Large number to give enough time to complete. Tasks terminate early when retraining finishes so this is not a problem.
NUM_TASKS=108
INFERENCE_CHUNKS=108
NUM_GPUS=1
EPOCHS=30
START_TASK=0
TERMINATION_TASK=72
MAX_INFERENCE_RESOURCES=0.25

# CityScapes cities we are interested in:
# All possible retraining hyperparameters

for HPARAM_ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17; do
    for CITY in las_vegas_24h_0_0.2fps las_vegas_24h_1_0.2fps las_vegas_24h_2_0.2fps las_vegas_24h_3_0.2fps; do
        echo Running profiling on city $CITY Hparam: $HPARAM_ID
        CUDA_VISIBLE_DEVICES=3 python driver_multicity.py \
                 --scheduler profiling \
                 --cities ${CITY} \
                 --log-dir ${RESULTS_ROOT}/${DATASET_NAME}_outputs/human_label_logs/ \
                 --retraining-period ${RETRAINING_PERIOD} \
                 --num-tasks ${NUM_TASKS} \
                 --inference-chunks ${INFERENCE_CHUNKS} \
                 --num-gpus ${NUM_GPUS} \
                 --dataset-name ${DATASET_NAME} \
                 --root ${DATASET_PATH} \
                 --restore-path ${MODEL_PATH} \
                 --hyperparameter-id $HPARAM_ID \
                 --start-task ${START_TASK} \
                 --termination-task ${TERMINATION_TASK} \
                 --epochs ${EPOCHS} \
                 --hyps-path ${HYPS_PATH} \
                 --inference-profile-path ${INFERENCE_PROFILE_PATH} \
                 --max-inference-resources ${MAX_INFERENCE_RESOURCES} \
                 --profiling-mode \
                 --profile-write-path ${RESULTS_ROOT}/${DATASET_NAME}_outputs/human_label_profiles_sampled_${START_TASK}_${TERMINATION_TASK}/ \
                 --golden-model-ckpt-path ${GOLDEN_MODEL_PATH} \
                 --use-data-cache \
                 --checkpoint-path ${RESULTS_ROOT}/${DATASET_NAME}_outputs/human_label_profiles_sampled_${START_TASK}_${TERMINATION_TASK}/ \
                 --gpu-memory 32
  done
done
