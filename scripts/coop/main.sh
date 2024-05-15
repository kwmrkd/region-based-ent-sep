#!/bin/bash

# custom config
DATA=datasets
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
# Officehome

LR=8e-5
GRID=8
K=90
WEIGHT=0.3

DOMAIN=art
for SEED in 1 2 3
do
        DIR=output_all/${DATASET}/${DOMAIN}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${GRID}/${LR}/${thres}/${WEIGHT}/seed${SEED}
        if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --source-domains ${DOMAIN} \
                --target-domains ${DOMAIN} \
                --trainer ${TRAINER} \
                --flag 3 \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --weight ${WEIGHT} \
                --k ${K} \
                --grid ${GRID} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                OPTIM.LR ${LR} \
                OPTIM.WARMUP_CONS_LR ${LR}
        fi
done

DOMAIN=clipart
for SEED in 1 2 3
do
        DIR=output_all/${DATASET}/${DOMAIN}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${GRID}/${LR}/${thres}/${WEIGHT}/seed${SEED}
        if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --source-domains ${DOMAIN} \
                --target-domains ${DOMAIN} \
                --trainer ${TRAINER} \
                --flag 1 \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --weight ${WEIGHT} \
                --k ${K} \
                --grid ${GRID} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                OPTIM.LR ${LR} \
                OPTIM.WARMUP_CONS_LR ${LR}
        fi
done

DOMAIN=product
for SEED in 1 2 3
do
        DIR=output_all/${DATASET}/${DOMAIN}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${GRID}/${LR}/${thres}/${WEIGHT}/seed${SEED}
        if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --source-domains ${DOMAIN} \
                --target-domains ${DOMAIN} \
                --trainer ${TRAINER} \
                --flag 1 \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --weight ${WEIGHT} \
                --k ${K} \
                --grid ${GRID} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                OPTIM.LR ${LR} \
                OPTIM.WARMUP_CONS_LR ${LR}
        fi
done

DOMAIN=real_world
for SEED in 1 2 3
do
        DIR=output_all/${DATASET}/${DOMAIN}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${GRID}/${LR}/${thres}/${WEIGHT}/seed${SEED}
        if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --source-domains ${DOMAIN} \
                --target-domains ${DOMAIN} \
                --trainer ${TRAINER} \
                --flag 0 \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --weight ${WEIGHT} \
                --k ${K} \
                --grid ${GRID} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                OPTIM.LR ${LR} \
                OPTIM.WARMUP_CONS_LR ${LR}
        fi
done

