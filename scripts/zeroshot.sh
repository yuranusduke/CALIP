#!/bin/bash

# custom config
DATA=./data/
TRAINER=CALIP_PF
DATASET=$1
CFG=rn50_ep200_ctxv1  # rn50

python main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${CFG}.yaml \
--output-dir output/${DATASET}/${TRAINER}/zero_shot/ \
--eval-only