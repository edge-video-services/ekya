#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='/home/researcher/datasets/cityscapes/'
MODEL_PATH='/home/researcher/models/'
#DATASET_PATH='/home/romilb/datasets/cityscapes_raw/'
#MODEL_PATH='/home/romilb/research/msr/models/'
HYPS_PATH='profiling/hyp_map_all.json'
INFERENCE_PROFILE_PATH='real_inference_profiles.csv'
RETRAINING_PERIOD=9999999 # Large number to give enough time to complete. Tasks terminate early when retraining finishes so this is not a problem.
NUM_TASKS=10
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=30
START_TASK=1
TERMINATION_TASK=9
MAX_INFERENCE_RESOURCES=0.25
DATASET_NAME=cityscapes

# Reduce number of epochs when using past retraining data
for CITY in jena zurich dusseldorf aachen bochum bremen darmstadt cologne monchengladbach stuttgart tubingen; do #zurich jena cologne; do
  for HPARAM_ID in 0 1 2 3 4 5 6; do # 7 8 9 10 11 12 13 14 15 16 17;
    echo Running profiling on city $CITY Hparam: $HPARAM_ID
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
  done
done