#!/bin/sh
DATASET_DIR="../dataset/2d_image_128"
EXPER_NAME="espcn_20k_image_2x"

python train.py \
    --dataset_dir ${DATASET_DIR} \
    --exper_name ${EXPER_NAME} \
    