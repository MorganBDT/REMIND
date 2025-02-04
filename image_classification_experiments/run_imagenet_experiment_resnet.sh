#!/usr/bin/env bash

PROJ_ROOT=/home/morgan/projects/REMIND

export PYTHONPATH=${PROJ_ROOT}

cd ${PROJ_ROOT}/image_classification_experiments

IMAGE_DIR=/media/data/Datasets/ImageNet2012
EXPT_NAME=remind_imagenet_resnet_scaleddown_withprinting
GPU="${1:-0}" # Default 0, include alternative GPU index as 1st argument to this script

REPLAY_SAMPLES=50
#MAX_BUFFER_SIZE=959665
MAX_BUFFER_SIZE=1200
CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
BASE_INIT_CLASSES=100
#CLASS_INCREMENT=100
CLASS_INCREMENT=2
#NUM_CLASSES=1000
NUM_CLASSES=104
BASE_INIT_CKPT=./imagenet_files/best_ResNet18ClassifyAfterLayer4_1_100.pth # base init ckpt file
LABEL_ORDER_DIR=./imagenet_files/ # location of numpy label files

CUDA_VISIBLE_DEVICES=${GPU} python -u -m memory_profiler imagenet_experiment.py \
--classifier ResNet18_StartAt_Layer4_1 \
--classifier_ckpt ${BASE_INIT_CKPT} \
--images_dir ${IMAGE_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${BASE_INIT_CLASSES} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--start_lr 0.1 \
--end_lr 0.001 \
--lr_step_size 100 \
--lr_mode step_lr_per_class \
--weight_decay 1e-5 \
--use_random_resized_crops \
--use_mixup \
--mixup_alpha .1 \
--label_dir ${LABEL_ORDER_DIR} \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log
