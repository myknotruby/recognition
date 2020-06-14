#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_GPU_WORKER_NTHREADS=3
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_BACKWARD_DO_MIRROR=1
export MXNET_GPU_MEM_POOL_RESERVE=5

#DATA_DIR=/train/mxnet-train/datasets/faces_emore

DATASET='rlag'

NETWORK=attse152
LR=0.01
BATCHSIZE=128
JOB=arcface-$DATASET
MODELDIR="./models/$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"

#CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='240000,400000' --models-root $MODELDIR --finetune '/train/mxnet-train/face/insightface-new/recognition/models/attse152-arcface-rl-0.01-24-new/models,41' #--exclude-layer-prefix 'fc7' #>"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='240000,400000' --models-root $MODELDIR --finetune '/train/mxnet-train/face/insightface-new/recognition/models/attse152-arcface-rl-0.01-24-new/models,41' --exclude-layer-prefix 'fc7' >"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='240000,400000' --models-root $MODELDIR --pretrained '/train/mxnet-train/face/insightface-new/recognition/models/attse152-arcface-rl-0.01-24-new/models,41' >"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='80000,140000' --models-root $MODELDIR >"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='0' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='60000,90000' --models-root $MODELDIR >"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='100000,240000,360000,450000' --models-root $MODELDIR >"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='100000,240000,360000,450000' --models-root $MODELDIR >"$LOGFILE-$DATASET-1" 2>&1 &
#CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr 0.01 --lr-steps='100000,240000,360000,450000' --finetune '/train/mxnet-train/face/insightface-new/recognition/models/attse152-arcface-rl-0.01-24-new/models,41' --exclude-layer-prefix 'fc2'  --models-root $MODELDIR >"$LOGFILE-$DATASET-2" 2>&1 &
#CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr 0.001 --lr-steps='100000,160000' --pretrained '/train/mxnet-train/face/insightface/recognition/models/attse152-arcface-rlag-0.1-64/attse152-arcface-rlag/model2,0' --exclude-layer-prefix ''  --models-root $MODELDIR >"$LOGFILE-$DATASET-2" 2>&1 &
#CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr 0.01 --lr-steps='100000,180000,260000' --pretrained '/train/mxnet-train/face/insightface/recognition/models/attse152-arcface-rlag-0.1-64/attse152-arcface-rlag/model2,0' --exclude-layer-prefix ''  --models-root $MODELDIR >"$LOGFILE-$DATASET-2" 2>&1 &
CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7' python -u train_parall.py --dataset=$DATASET --network "$NETWORK" --per-batch-size $BATCHSIZE --lr $LR --lr-steps='40000,80000' --finetune '/train/mxnet-train/face/insightface/recognition/models/attse152-arcface-rlag-0.1-64/attse152-arcface-rlag/model2,0' --exclude-layer-prefix 'fc2,fc7'  --models-root $MODELDIR >"$LOGFILE-$DATASET-3" 2>&1 &
