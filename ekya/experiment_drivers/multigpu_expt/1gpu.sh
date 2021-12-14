#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='/ekya/datasets/cityscapes/'
MODEL_PATH='/ekya/models/'
HYPS_PATH='/ekya/ekya/ekya/experiment_drivers/utilitysim_schedules/3city_0713_sysprof_fixedseed/hyp_map.json'
INFERENCE_PROFILE_PATH='/ekya/ekya/ekya/experiment_drivers/real_inference_profiles.csv'
LOG_BASE_DIR='/ekya/expt_results/multicity'
#UTILITYSIM_SCHEDULE_KEY='100_1_thief_True'
RETRAINING_PERIOD=200
NUM_TASKS=10
INFERENCE_CHUNKS=10
EPOCHS=15  # Matters for fair scheduler
START_TASK=1
TERMINATION_TASK=5  # TODO: Change this
MAX_INFERENCE_RESOURCES=0.25
CITIES=zurich,jena #,cologne
DEFAULT_PARAM_ID=5
MP_RES_PER_TRIAL=0.25
DATASET_NAME=cityscapes

NUM_GPUS=1
GPU_MEMORY=15

# Run for 8,4,3 cities
CITIES_10=jena,zurich,darmstadt,aachen,bremen,cologne,dusseldorf,monchengladbach,stuttgart,tubingen #bochum
CITIES_8=jena,zurich,darmstadt,aachen,bremen,cologne,dusseldorf,monchengladbach #,stuttgart,tubingen #bochum
CITIES_6=jena,zurich,darmstadt,aachen,bremen,cologne # ,dusseldorf,monchengladbach # ,stuttgart,tubingen #bochum
CITIES_4=jena,zurich,darmstadt,aachen #,bremen,cologne,dusseldorf,monchengladbach #,stuttgart,tubingen #bochum
CITIES_2=jena,zurich #,darmstadt,aachen #,bremen,cologne,dusseldorf,monchengladbach #,stuttgart,tubingen #bochum
for CITIES in ${CITIES_2} ${CITIES_6}; do # ${CITIES_10}
  # Fair scheduler
  SCHEDULER=fair
  for INFERENCE_WEIGHT in 0.5; do # 1 0.25 0.75
    NUM_CITIES=$(echo "${CITIES}" | awk -F "," ' { print NF } ')
    echo Running fair scheduler on cities ${CITIES} with weight ${INFERENCE_WEIGHT}
    sleep 5
    python ../driver_multicity.py --scheduler ${SCHEDULER} \
             --fair-inference-weight ${INFERENCE_WEIGHT} \
             --cities ${CITIES} \
             --log-dir ${LOG_BASE_DIR}/${NUM_GPUS}gpu/${NUM_CITIES}cities/${SCHEDULER}_${INFERENCE_WEIGHT} \
             --retraining-period ${RETRAINING_PERIOD} \
             --num-tasks ${NUM_TASKS} \
             --inference-chunks ${INFERENCE_CHUNKS} \
             --num-gpus ${NUM_GPUS} \
             --gpu-memory ${GPU_MEMORY} \
             --dataset-name ${DATASET_NAME} \
             --root ${DATASET_PATH} \
             --use-data-cache \
             --restore-path ${MODEL_PATH} \
             --lists-pretrained frankfurt,munster \
             --hyperparameter-id ${DEFAULT_PARAM_ID} \
             --start-task ${START_TASK} \
             --termination-task ${TERMINATION_TASK} \
             --epochs ${EPOCHS} \
             --hyps-path ${HYPS_PATH} \
             --inference-profile-path ${INFERENCE_PROFILE_PATH} \
             --max-inference-resources ${MAX_INFERENCE_RESOURCES} \
             --microprofile-resources-per-trial ${MP_RES_PER_TRIAL}
  done

  # Thief scheduler
  SCHEDULER=thief
  NUM_CITIES=$(echo "${CITIES}" | awk -F "," ' { print NF } ')
  echo Running scheduler ${SCHEDULER} on cities ${CITIES}
  sleep 5
  python ../driver_multicity.py --scheduler ${SCHEDULER} \
           --cities ${CITIES} \
           --log-dir ${LOG_BASE_DIR}/${NUM_GPUS}gpu/${NUM_CITIES}cities/${SCHEDULER} \
           --retraining-period ${RETRAINING_PERIOD} \
           --num-tasks ${NUM_TASKS} \
           --inference-chunks ${INFERENCE_CHUNKS} \
           --num-gpus ${NUM_GPUS} \
           --gpu-memory ${GPU_MEMORY} \
           --dataset-name ${DATASET_NAME} \
           --root ${DATASET_PATH} \
           --use-data-cache \
           --restore-path ${MODEL_PATH} \
           --lists-pretrained frankfurt,munster \
           --hyperparameter-id ${DEFAULT_PARAM_ID} \
           --start-task ${START_TASK} \
           --termination-task ${TERMINATION_TASK} \
           --epochs ${EPOCHS} \
           --hyps-path ${HYPS_PATH} \
           --inference-profile-path ${INFERENCE_PROFILE_PATH}   \
           --max-inference-resources ${MAX_INFERENCE_RESOURCES} \
           --microprofile-resources-per-trial ${MP_RES_PER_TRIAL}
done