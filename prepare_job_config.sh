#!/usr/bin/env bash
DATASET_PATH="/tmp/dataset/data.csv"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved Breast Cancer dataset in ${DATASET_PATH}"
fi

# For FL dataset(1/3 Test, 1/3 TrainA, 1/3 TrainB):
valid_frac=0.3333333333

# For local dataset (10%test, 10% train)
# valid_frac=0.1

echo "Generating job configs with data splits, reading from ${DATASET_PATH}"

task_name="sklearn_svm"
for site_num in 2;
do
    for split_mode in uniform;
    do
        python3 utils/prepare_job_config.py \
        --task_name "${task_name}" \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --valid_frac ${valid_frac} \
        --split_method ${split_mode}
    done
done
