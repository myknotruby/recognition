#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=3
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_BACKWARD_DO_MIRROR=1
export MXNET_GPU_MEM_POOL_RESERVE=5

#DATA_DIR=/train/mxnet-train/datasets/faces_emore

DATASET='retina'

#NETWORK=r152
NETWORK=r100
JOB=arcface-$DATASET
MODELDIR="./models/$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"

#CUDA_VISIBLE_DEVICES='2,3,4,5,6,7' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 64 --lr=0.01 >"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='0' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 8 --lr=0.01
#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 20 --exclude-layer-prefix "fc" --lr=0.1  >"$LOGFILE-$DATASET-001" 2>&1 &
#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --pretrained "$PREFIX,87" --per-batch-size 20 --exclude-layer-prefix "fc" --lr=0.01 >"$LOGFILE-$DATASET-02" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --finetune "$PREFIX,0" --per-batch-size 160 --lr-steps='100000,200000,300000,350000' --exclude-layer-prefix 'fc' --lr 0.1 >"$LOGFILE-$DATASET-00" 2>&1 &
#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --finetune "$PREFIX,99" --per-batch-size 128 --exclude-layer-prefix "fc2,fc7" --lr=0.1 >"$LOGFILE-$DATASET-01" 2>&1 &
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --pretrained "$PREFIX,104" --per-batch-size 20 --exclude-layer-prefix "fc2,fc7" --lr=0.01 >"$LOGFILE-$DATASET-03" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --pretrained "$PREFIX,104" --per-batch-size 20 --exclude-layer-prefix "fc2,fc7" --lr=0.1 >"$LOGFILE-$DATASET-3" 2>&1 &
CUDA_VISIBLE_DEVICES='0' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 128 --lr-steps='100000,200000,300000,350000' --exclude-layer-prefix 'fc' --lr 0.1