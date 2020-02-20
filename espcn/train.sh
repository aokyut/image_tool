#!/bin/sh
DATASET_DIR="../dataset/2d_image_128_20k"
EXPER_NAME="test_rap1"
RESUME_STEP=0

python train.py \
    --dataset_dir ${DATASET_DIR} \
    --exper_name ${EXPER_NAME} \
    --resume_step ${RESUME_STEP} \
    --n_save_model 10000 \
    --epoch 20 \
    --n_record_iter 100 \
    --batch_size 64 \
    --gpu \
