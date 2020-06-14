#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=3
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_BACKWARD_DO_MIRROR=1
export MXNET_GPU_MEM_POOL_RESERVE=5

#DATA_DIR=/train/mxnet-train/datasets/faces_emore

DATASET='emore'
#DATASET='ela'

#NETWORK=r152
NETWORK=eb7
JOB=arcface-$DATASET
MODELDIR="./models/$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"


#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 28 --exclude-layer-prefix 'fc' --lr 0.1 --finetune "$PREFIX,1" >"$LOGFILE-$DATASET-01" 2>&1 &
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u _train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 28 --exclude-layer-prefix 'fc' --lr 0.1 --pretrained "$PREFIX,1" >"$LOGFILE-$DATASET-01" 2>&1 &
#CUDA_VISIBLE_DEVICES='1,2,3' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 64 --lr-steps='200000,250000,300000,350000' --pretrained "$PREFIX,2" --exclude-layer-prefix 'fc' --lr 0.001 >"$LOGFILE-$DATASET-00" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --finetune "$PREFIX,0" --per-batch-size 128 --lr-steps='20000,40000,60000' --exclude-layer-prefix "fc" --lr 0.1
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --pretrained "$PREFIX,0"  --per-batch-size 32 --exclude-layer-prefix "fc"--lr-steps='100000,200000,300000,400000' >"$LOGFILE-$DATASET-00" 2>&1 &

#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --pretrained "$PREFIX,72"  --per-batch-size 64 --exclude-layer-prefix "fc" --lr=0.001 --lr-steps='150000,300000,400000' >"$LOGFILE-$DATASET-01" 2>&1 &

#liaohuan edit
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --finetune "$PREFIX,122"  --per-batch-size 256 --exclude-layer-prefix "fc" --lr=0.1 --lr-steps='30000,60000,100000' >"$LOGFILE-$DATASET-01" 2>&1 &
#CUDA_VISIBLE_DEVICES='2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --pretrained "$PREFIX,12"  --per-batch-size 64 --exclude-layer-prefix "fc" --lr=0.1 --lr-steps='160000,320000,480000,560000' >"$LOGFILE-$DATASET-01" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 32 --pretrained "$PREFIX,43" --lr=0.001 --lr-steps='160000,350000,500000' >"$LOGFILE-$DATASET-002" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size 32 --lr=0.01 --lr-steps='160000,350000,500000' >"$LOGFILE-$DATASET-001" 2>&1 &


#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --finetune "$PREFIX,10"  --per-batch-size 256 --exclude-layer-prefix "fc" --lr=0.1 --lr-steps='30000,60000,100000' >"$LOGFILE-$DATASET-01" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --pretrained "$PREFIX,69"  --per-batch-size 64 --exclude-layer-prefix "fc" --lr=0.001 --lr-steps='60000,260000,480000' >"$LOGFILE-$DATASET-004" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --pretrained "$PREFIX,1"  --per-batch-size 32 --exclude-layer-prefix "fc" --lr=0.01 --lr-steps='60000,260000,480000' >"$LOGFILE-$DATASET-001" 2>&1 &
#CUDA_VISIBLE_DEVICES='4' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --per-batch-size 32 --lr=0.01 --lr-steps='60000,260000,480000'
