#!/bin/bash
set -e

NUM_TASKS=10
# DATASET_NAME='cityscapes'
DATASET_NAME='waymo'
MODEL_NAME='resnet18'
HYPS_PATH='./hyp_map_all.json'
if [ ${DATASET_NAME} = 'cityscapes' ]
then
    echo "cityscapes"
    DATASET_PATH='../../datasets/CityScapes'
    CITIES=" bochum bremen cologne darmstadt dusseldorf jena monchengladbach stuttgart tubingen zurich"

    SAVE_PATH='./results_golden_label_cityscapes'
elif [ ${DATASET_NAME} = 'waymo' ]
then
    echo 'waymo'
    DATASET_PATH='../../datasets/waymo/waymo_classification_images'
    CITIES="sf_000_009 sf_020_029 sf_030_039 sf_050_059 sf_060_069 sf_070_079 sf_080_089"

    SAVE_PATH='./results_golden_label_waymo'
    CACHED_MODEL_PATH='../../results/model_cache_exp/golden_label_profiles'
else
    echo 'Dataset type not supported!'
fi

echo ${DATASET_PATH}


for CITY in ${CITIES}; do
    echo ${CITY}
    CUDA_VISIBLE_DEVICES=2 python run.py \
        --dataset ${DATASET_NAME} \
        --root ${DATASET_PATH} \
        --camera_name ${CITY} \
        --num_tasks ${NUM_TASKS} \
        --checkpoint_path ${CACHED_MODEL_PATH} \
        --save_path ${SAVE_PATH} \
        --hyp_map_path ${HYPS_PATH}
done
