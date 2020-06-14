#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=3
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_BACKWARD_DO_MIRROR=1
export MXNET_GPU_MEM_POOL_RESERVE=5

#DATA_DIR=/train/mxnet-train/datasets/faces_emore

DATASET='africa'

NETWORK=mv3
JOB=arcface-$DATASET
MODELDIR="./models/$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model2"
LOGFILE="$MODELDIR/log"


#CUDA_VISIBLE_DEVICES='4' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 1 --lr-steps='200000,320000,350000' --pretrained "$PREFIX,73" --exclude-layer-prefix 'fc' --lr 0.001
#CUDA_VISIBLE_DEVICES='6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 100 --lr-steps='100000,200000,300000,350000' --exclude-layer-prefix 'fc' --lr 0.001 >"$LOGFILE-$DATASET-1" 2>&1 &
CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 112 --lr-steps='100000,150000' --pretrained "$PREFIX,90"  --lr 0.001 >"$LOGFILE-$DATASET-t1-21" 2>&1 &
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 120 --lr-steps='200000,320000,350000' --lr 0.001 >"$LOGFILE-$DATASET-t1" 2>&1 &
