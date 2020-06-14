#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=3
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_BACKWARD_DO_MIRROR=1
export MXNET_GPU_MEM_POOL_RESERVE=5

#DATA_DIR=/train/mxnet-train/datasets/faces_emore

DATASET='casia'

#NETWORK=r152
NETWORK=r50
JOB=arcface-$DATASET-E
MODELDIR="./models/$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"


#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --finetune "$PREFIX,0" --per-batch-size 160 --lr-steps='100000,200000,300000,350000' --exclude-layer-prefix 'fc' --lr 0.1 >"$LOGFILE-$DATASET-00" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --finetune "$PREFIX,0" --per-batch-size 128 --lr-steps='20000,40000,60000' --exclude-layer-prefix "fc" --lr 0.1
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --pretrained "$PREFIX,0"  --per-batch-size 32 --exclude-layer-prefix "fc"--lr-steps='100000,200000,300000,400000' >"$LOGFILE-$DATASET-00" 2>&1 &

#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --pretrained "$PREFIX,72"  --per-batch-size 64 --exclude-layer-prefix "fc" --lr=0.001 --lr-steps='150000,300000,400000' >"$LOGFILE-$DATASET-01" 2>&1 &

#liaohuan edit

#e
CUDA_VISIBLE_DEVICES='2,3' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --models-root=$MODELDIR --finetune "/train/mxnet-train/face/insightface/recognition/models/r50-arcface-egl/ecuador20180717/model,6"  --per-batch-size 96 --lr=0.01 --lr-steps='30000,60000,80000' >"$LOGFILE-$DATASET-002" 2>&1 &
#edul
#CUDA_VISIBLE_DEVICES='6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --models-root=$MODELDIR --finetune "/train/mxnet-train/face/insightface/recognition/models/r50-arcface-egl/ecuador20180717/model,6"  --per-batch-size 96 --lr=0.01 --lr-steps='30000,60000,80000' >"$LOGFILE-$DATASET-002" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --models-root=$MODELDIR --finetune "/train/mxnet-train/face/insightface/recognition/models/r50-arcface-egl/ecuador20180717/model,6"  --per-batch-size 96 --lr=0.01 --lr-steps='30000,60000,80000' >"$LOGFILE-$DATASET-drop-001" 2>&1 &
#CUDA_VISIBLE_DEVICES='6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK"  --per-batch-size 64 --lr=0.1 --lr-steps='60000,100000,140000'
