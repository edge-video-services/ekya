#!/bin/bash
set -e

PATH_TO_MODULE=/ekya/ekya/ekya/
DATASET_NAME='cityscapes'
# DATASET_NAME='waymo'
# DATASET_NAME='mp4'
HYPS_PATH='/ekya/ekya/ekya/experiment_drivers/profiling/hyp_map_all.json'
INFERENCE_PFOFILE_PATH='/ekya/ekya/ekya/experiment_drivers/real_inference_profiles.csv'
RETRAINING_PERIOD="200"

# Parameters for the fair fixed scheduler
FAIRFIXED_CONFIG_ID=4
FAIRFIXED_CONFIG_EPOCHS=17

# Cloud scheduler params
CLOUD_DELAY=221,364,100,75

if [ ${DATASET_NAME} = 'cityscapes' ]
then
    # cityscapes cities: aachen bochum bremen cologne darmstadt dusseldorf monchengladbach stuttgart tubingen zurich
    # PROFILE_ROOT='/data2/zxxia/ekya/ekya/experiment_drivers/cityscapes_outputs/profiles_all_hyper'  # human
    PROFILE_ROOT='/ekya/results/profiling/nsdi/cityscapes_outputs/golden_label_profiles'  # godel model
    CITIES="aachen bochum bremen cologne darmstadt dusseldorf monchengladbach stuttgart tubingen zurich" # jena
    GOLDEN_MODEL_DELAY=2
    NUM_TASKS=10
    OUTPUT_PATH="/ekya/results/profiling/nsdi/cityscapes_outputs/cloud/results_allcloud_0916"
    # OUTPUT_PATH='/data2/zxxia/ekya/tests'
elif [ ${DATASET_NAME} = 'waymo' ]
then
    # Waymo cities: phx_020_029  phx_030_039  phx_040_049  phx_050_059  sf_000_009  sf_020_029  sf_030_039  sf_050_059  sf_060_069  sf_070_079
    PROFILE_ROOT='/data2/zxxia/ekya/results/waymo_outputs/golden_label_profiles'  # godel model
    CITIES='phx_020_029  phx_030_039  phx_040_049  phx_050_059  sf_000_009  sf_020_029  sf_030_039  sf_050_059  sf_060_069  sf_070_079'
    GOLDEN_MODEL_DELAY=2 # TODO: need to measure the delay of waymo dataset
    NUM_TASKS=10
    OUTPUT_PATH='/data2/zxxia/ekya/results/waymo_outputs/golden_label_profiles'
elif [ ${DATASET_NAME} = 'mp4' ]
then
    PROFILE_ROOT='/data2/zxxia/ekya/results/mp4_outputs/human_label_profiles_pre'
    # las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good \
    CITIES='las_vegas_2fps_7201_108000 las_vegas_2fps_7201_108000 las_vegas_2fps_7201_108000 las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2fps_very_good las_vegas_2_2fps las_vegas_2_2fps las_vegas_1_2fps_bad las_vegas_1_2fps_bad'
    GOLDEN_MODEL_DELAY=2 # TODO: need to measure the delay of mp4 dataset
    NUM_TASKS=18
    OUTPUT_PATH='/data2/zxxia/ekya/tests/mp4/mixed'
fi

python ${PATH_TO_MODULE}/simulation \
    --dataset ${DATASET_NAME} \
    --root ${PROFILE_ROOT} \
    --camera_names ${CITIES} \
    --retraining_periods ${RETRAINING_PERIOD}  \
    --delay ${GOLDEN_MODEL_DELAY} \
    --num_tasks ${NUM_TASKS} \
    --output_path  ${OUTPUT_PATH} \
    --hyp_map_path ${HYPS_PATH} \
    --hyperparameters 0 1 2 3 4 5 \
    --real_inference_profiles ${INFERENCE_PFOFILE_PATH} \
    --fairfixed_config_id ${FAIRFIXED_CONFIG_ID} \
    --fairfixed_config_epochs ${FAIRFIXED_CONFIG_EPOCHS} \
    --cloud_delay ${CLOUD_DELAY}
