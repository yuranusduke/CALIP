#!/bin/bash

# custom config
DATA=./data/
TRAINER=CALIP_FS

DATASET=$1
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)

CFG=rn50_ep200_ctxv1
SEED=2

#for SEED in 1 2 3
#do
DIR=output/${DATASET}/${TRAINER}/${SHOTS}_shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainer/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
#done