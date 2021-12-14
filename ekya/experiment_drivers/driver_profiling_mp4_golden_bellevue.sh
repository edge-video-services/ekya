#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure
# retraining results in accuracy gains.
set -e

DATASET_NAME=bellevue
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
    # for CITY in Bellevue_150th_Eastgate__2017-09-10_18-08-24_0.5fps \
    #     Bellevue_150th_Eastgate__2017-09-10_21-08-28_0.5fps \
    #     Bellevue_150th_Eastgate__2017-09-11_00-08-29_0.5fps \
    #     Bellevue_150th_Eastgate__2017-09-11_03-08-30_0.5fps \
    #     Bellevue_150th_Eastgate__2017-09-11_06-08-30_0.5fps \
    #     Bellevue_150th_Eastgate__2017-09-11_09-08-31_0.5fps \
    #     Bellevue_150th_Eastgate__2017-09-11_14-08-31_0.5fps \
    #     Bellevue_150th_Eastgate__2017-09-11_15-08-33_0.5fps; do
    # for CITY in Bellevue_150th_Newport__2017-09-10_18-08-24_0.5fps \
    #     Bellevue_150th_Newport__2017-09-10_21-08-28_0.5fps \
    #     Bellevue_150th_Newport__2017-09-11_00-08-29_0.5fps \
    #     Bellevue_150th_Newport__2017-09-11_03-08-29_0.5fps \
    #     Bellevue_150th_Newport__2017-09-11_06-08-30_0.5fps \
    #     Bellevue_150th_Newport__2017-09-11_09-08-30_0.5fps \
    #     Bellevue_150th_Newport__2017-09-11_12-08-32_0.5fps \
    #     Bellevue_150th_Newport__2017-09-11_15-08-32_0.5fps; do
    # for CITY in Bellevue_150th_SE38th__2017-09-10_18-08-24_0.5fps \
    #     Bellevue_150th_SE38th__2017-09-10_21-08-38_0.5fps \
    #     Bellevue_150th_SE38th__2017-09-11_00-08-29_0.5fps \
    #     Bellevue_150th_SE38th__2017-09-11_03-08-30_0.5fps \
    #     Bellevue_150th_SE38th__2017-09-11_06-08-32_0.5fps \
    #     Bellevue_150th_SE38th__2017-09-11_09-08-31_0.5fps \
    #     Bellevue_150th_SE38th__2017-09-11_12-08-38_0.5fps \
    #     Bellevue_150th_SE38th__2017-09-11_16-08-35_0.5fps; do
    # for CITY in Bellevue_Bellevue_NE8th__2017-09-10_18-08-23_0.5fps \
    #     Bellevue_Bellevue_NE8th__2017-09-10_21-08-28_0.5fps \
    #     Bellevue_Bellevue_NE8th__2017-09-11_00-08-29_0.5fps \
    #     Bellevue_Bellevue_NE8th__2017-09-11_03-08-29_0.5fps \
    #     Bellevue_Bellevue_NE8th__2017-09-11_06-08-29_0.5fps \
    #     Bellevue_Bellevue_NE8th__2017-09-11_09-08-30_0.5fps \
    #     Bellevue_Bellevue_NE8th__2017-09-11_12-08-31_0.5fps \
    #     Bellevue_Bellevue_NE8th__2017-09-11_15-08-32_0.5fps; do
    for CITY in Bellevue_116th_NE12th__2017-09-10_19-08-25_0.5fps \
        Bellevue_116th_NE12th__2017-09-10_22-08-50_0.5fps \
        Bellevue_116th_NE12th__2017-09-11_01-08-29_0.5fps \
        Bellevue_116th_NE12th__2017-09-11_04-08-30_0.5fps \
        Bellevue_116th_NE12th__2017-09-11_07-08-32_0.5fps \
        Bellevue_116th_NE12th__2017-09-11_11-08-33_0.5fps \
        Bellevue_116th_NE12th__2017-09-11_15-08-36_0.5fps \
        Bellevue_116th_NE12th__2017-09-11_16-08-37_0.5fps; do
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
