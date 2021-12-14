#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='/home/researcher/datasets/cityscapes/'
MODEL_PATH='/home/researcher/models/'
#DATASET_PATH='/home/romilb/datasets/cityscapes_raw/'
#MODEL_PATH='/home/romilb/research/msr/models/'
HYPS_PATH='utilitysim_schedules/3city_0707_sysprof_only18/hyp_map.json'
INFERENCE_PROFILE_PATH='real_inference_profiles.csv'
RETRAINING_PERIOD=100
NUM_TASKS=10
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=1
START_TASK=3
TERMINATION_TASK=1
MAX_INFERENCE_RESOURCES=0.25
CITIES=jena
HP_ID=0

# Reduce number of epochs when using past retraining data
python driver_noekya.py \
           --cities ${CITIES} \
           --log-dir /tmp/ekya_expts/multicity/${UTILITYSIM_SCHEDULE_KEY}/ \
           --retraining-period ${RETRAINING_PERIOD} \
           --num-tasks ${NUM_TASKS} \
           --inference-chunks ${INFERENCE_CHUNKS} \
           --num-gpus ${NUM_GPUS} \
           --root ${DATASET_PATH} \
           --use-data-cache \
           --restore-path ${MODEL_PATH} \
           --lists-pretrained frankfurt,munster \
           --hyperparameter-id $HP_ID \
           --start-task ${START_TASK} \
           --termination-task ${TERMINATION_TASK} \
           --epochs ${EPOCHS} \
           --hyps-path ${HYPS_PATH} \
           --inference-profile-path ${INFERENCE_PROFILE_PATH} \
           --max-inference-resources ${MAX_INFERENCE_RESOURCES}