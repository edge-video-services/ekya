#!/bin/bash
# Runs one city at a time through ekya but for multiple hyperparameters. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='/home/researcher/datasets/cityscapes/'
MODEL_PATH='/home/researcher/models/'
RETRAINING_PERIOD=360
NUM_TASKS=5
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=5
START_TASK=0
TERM_TASK=-1

# Reduce number of epochs when using past retraining data
for city in zurich stuttgart darmstadt dusseldorf monchengladbach aachen tubingen bochum bremen cologne ulm jena strasbourg hamburg krefeld weimar hanover erfurt; do
  for hyperparamid in 0 10 11; do
    for scheduler in fair; do
      echo Running scheduler $scheduler on city $city with hparam $hyperparamid
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
             --hyperparameter-id ${hyperparamid} \
             --start-task ${START_TASK} \
             --termination-task ${TERM_TASK} \
             --lists-pretrained frankfurt,munster
    done
  done

  for scheduler in noretrain; do
    python driver_singlecity.py --scheduler ${scheduler} \
             --cities ${city} \
             --log-dir /tmp/ekya_expts/singlecity/${scheduler}/${city} \
             --retraining-period ${RETRAINING_PERIOD} \
             --num-tasks ${NUM_TASKS} \
             --inference-chunks ${INFERENCE_CHUNKS} \
             --num-gpus ${NUM_GPUS} \
             --epochs ${EPOCHS} \
             --root ${DATASET_PATH} \
             --use-data-cache \
             --restore-path ${MODEL_PATH} \
             --hyperparameter-id 0 \
             --start-task ${START_TASK} \
             --lists-pretrained frankfurt,munster
  done
done