#!/usr/bin/env bash

GPU="${1:-0}" # Default 0, include alternative GPU index as 1st argument to this script
RUN="${2:-0}"

PROJ_ROOT=/home/mbt10/REMIND

export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}/image_classification_experiments

IMAGENET_DIR=/n/groups/kreiman/shared_data/Imagenet2012
BASE_MAX_CLASS=100
MODEL=SqueezeNetClassifyAfterLayer12
CKPT_FILE=SqueezeNetClassifyAfterLayer12_base_init_origpqreshape_run${RUN}.pth
LABEL_ORDER_DIR=./imagenet_files_run${RUN}/ # location of numpy label files

CUDA_VISIBLE_DEVICES=${GPU} python -u train_base_init_network_from_scratch.py \
--lr 0.001 \
--epochs 15 \
--arch ${MODEL} \
--ckpt_file ${CKPT_FILE} \
--data ${IMAGENET_DIR} \
--base_max_class ${BASE_MAX_CLASS} \
--labels_dir ${LABEL_ORDER_DIR} > logs/${MODEL}_${BASE_MAX_CLASS}_base_init_origpqreshape_run${RUN}.log
