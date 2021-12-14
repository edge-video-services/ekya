#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='/home/researcher/datasets/cityscapes/'
MODEL_PATH='/home/researcher/models/'
#DATASET_PATH='/home/romilb/datasets/cityscapes_raw/'
#MODEL_PATH='/home/romilb/research/msr/models/'
RETRAINING_PERIOD=360
NUM_TASKS=5
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=10
START_TASK=1

# Reduce number of epochs when using past retraining data
for city in zurich stuttgart darmstadt dusseldorf; do # monchengladbach aachen tubingen bochum bremen cologne ulm jena strasbourg hamburg krefeld weimar hanover erfurt; do
  for scheduler in fair noretrain; do
    echo Running scheduler $scheduler on city $city
    python driver_singlecity.py --scheduler ${scheduler} \
             --cities ${city} \
             --log-dir /tmp/ekya_expts/singlecity/${scheduler}_${hyperparamid}/${city} \
             --retraining-period ${RETRAINING_PERIOD} \
             --num-tasks ${NUM_TASKS} \
             --inference-chunks ${INFERENCE_CHUNKS} \
             --num-gpus ${NUM_GPUS} \
             --epochs ${EPOCHS} \
             --root ${DATASET_PATH} \
             --use-data-cache \
             --restore-path ${MODEL_PATH} \
             --lists-pretrained frankfurt,munster \
             --hyperparameter-id 0 \
             --start-task ${START_TASK}
  done
done
