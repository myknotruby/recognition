#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=3
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_BACKWARD_DO_MIRROR=1
export MXNET_GPU_MEM_POOL_RESERVE=5

DATASET='rl'

NETWORK=attse152
LR=0.1
BATCHSIZE=24
JOB=arcface-$DATASET-$LR-$BATCHSIZE
MODELDIR="./models/$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"


#CUDA_VISIBLE_DEVICES='0' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='150000,250000,350000,400000' --models-root $MODELDIR --finetune '/train/mxnet-train/face/insightface-new/recognition/models/attse152relu-arcface-rlag-0.001-24-new-61/model,71' --exclude-layer-prefix 'fc' >"$LOGFILE-$DATASET-2" 2>&1 &
CUDA_VISIBLE_DEVICES='0' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='150000,250000,350000,400000' --models-root $MODELDIR --pretrained '/train/mxnet-train/face/insightface/recognition/models/attse152-arcface-rl-0.1-24/attse152-arcface-rl/model,1' --exclude-layer-prefix 'fc1' #>"$LOGFILE-$DATASET-2" 2>&1 &
