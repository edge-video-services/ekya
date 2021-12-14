#!/bin/bash
# Runs a short profiling. Useful for accuracy sanity check across platforms
set -e

#DATASET_PATH='/home/researcher/datasets/cityscapes/'
#MODEL_PATH='/home/researcher/models/'
DATASET_PATH='/home/romilb/datasets/cityscapes_raw/'
MODEL_PATH='/home/romilb/research/msr/models/'
HYPS_PATH='profiling/hyp_map_all.json'
INFERENCE_PROFILE_PATH='real_inference_profiles.csv'
RETRAINING_PERIOD=9999999 # Large number to give enough time to complete. Tasks terminate early when retraining finishes so this is not a problem.
NUM_TASKS=10
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=10
START_TASK=1
TERMINATION_TASK=9
MAX_INFERENCE_RESOURCES=0.25
DATASET_NAME=cityscapes
CITY=jena
HPARAM_ID=0

echo Running platform sanity check on city $CITY Hparam: $HPARAM_ID
python driver_multicity.py \
         --scheduler profiling \
         --cities ${CITY} \
         --log-dir /tmp/ekya_expts/profiling/logs/ \
         --retraining-period ${RETRAINING_PERIOD} \
         --num-tasks ${NUM_TASKS} \
         --inference-chunks ${INFERENCE_CHUNKS} \
         --num-gpus ${NUM_GPUS} \
         --dataset-name ${DATASET_NAME} \
         --root ${DATASET_PATH} \
         --use-data-cache \
         --restore-path ${MODEL_PATH} \
         --lists-pretrained frankfurt,munster \
         --hyperparameter-id $HPARAM_ID \
         --start-task ${START_TASK} \
         --termination-task ${TERMINATION_TASK} \
         --epochs ${EPOCHS} \
         --hyps-path ${HYPS_PATH} \
         --inference-profile-path ${INFERENCE_PROFILE_PATH} \
         --max-inference-resources ${MAX_INFERENCE_RESOURCES} \
         --profiling-mode \
         --profile-write-path /tmp/ekya_expts/profiling/profiles/