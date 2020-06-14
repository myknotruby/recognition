#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=3
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_BACKWARD_DO_MIRROR=1
export MXNET_GPU_MEM_POOL_RESERVE=5

#DATA_DIR=/train/mxnet-train/datasets/faces_emore

DATASET='retina'

#NETWORK=r152
NETWORK=resnest
JOB=arcface-$DATASET
MODELDIR="./models/$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"

#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 42 --lr=0.001 --lr-steps='160000,320000,400000' --pretrained '/train/mxnet-train/face/insightface/recognition/models/resnest-arcface-retina/model2,7'  >"$LOGFILE-$DATASET-01" 2>&1 &
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 48 --lr=0.01 --lr-steps='160000,320000,400000'  >"$LOGFILE-$DATASET-04" 2>&1 &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 48 --lr=0.001 --lr-steps='160000,320000,400000' --pretrained '/train/mxnet-train/face/insightface/recognition/models/resnest-arcface-retina/model2,9'  >"$LOGFILE-$DATASET-03" 2>&1 &
#CUDA_VISIBLE_DEVICES='2,3' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 2 --lr=0.01 --lr-steps='160000,320000,400000'  >"$LOGFILE-$DATASET-01" 2>&1 &
