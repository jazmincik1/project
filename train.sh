#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 MODEL_NAME [MODEL_VERSION]"
    exit 1
fi

MODEL_NAME="$1"

MODEL_VERSION="${2:-'18'}"

if [ "$MODEL_NAME" == "resnet18_train" ]; then
    python scripts/resnet/resnet18_train.py \
        --run-name resnet_first \
        --dataset-dir /content/drive/MyDrive/colab_data/animals \
        --learning-rate 0.001 \
        --batch-size 16 \
        --num-workers 2 \
        --num-epochs 50 \
        --load-checkpoints 0 \
        --load-checkpoints-path /home \
        --save-checkpoints 1 \
        --save-checkpoints-epoch 5

elif [ "$MODEL_NAME" == "resnet_fine_tune" ]; then
    python scripts/resnet/resnet_fine_tune.py \
        --resnet-version "$MODEL_VERSION" \
        --run-name "$MODEL_NAME-$MODEL_VERSION-run" \
        --dataset-dir /content/drive/MyDrive/colab_data/animals \
        --learning-rate 0.001 \
        --batch-size 16 \
        --num-workers 2 \
        --num-epochs 50 \
        --load-checkpoints 0 \
        --load-checkpoints-path /home \
        --save-checkpoints 1 \
        --save-checkpoints-epoch 5

elif [ "$MODEL_NAME" == "lenet_train" ]; then
    python scripts/lenet/lenet_train.py \
        --run-name lenet_train \
        --dataset-dir /content/drive/MyDrive/colab_data/animals \
        --learning-rate 0.001 \
        --batch-size 16 \
        --num-workers 2 \
        --num-epochs 50 \
        --load-checkpoints 0 \
        --load-checkpoints-path /home \
        --save-checkpoints 1 \
        --save-checkpoints-epoch 5

elif [ "$MODEL_NAME" == "vgg_fine_tune" ]; then
    python scripts/vgg/vgg_fine_tune.py \
        --run-name vgg_fine_tune \
        --dataset-dir /content/drive/MyDrive/colab_data/animals \
        --learning-rate 0.001 \
        --batch-size 16 \
        --num-workers 2 \
        --num-epochs 50 \
        --load-checkpoints 0 \
        --load-checkpoints-path /home \
        --save-checkpoints 1 \
        --save-checkpoints-epoch 10 \
        --vgg-version 16

elif [ "$MODEL_NAME" == "randomforest_train" ]; then
    python scripts/forest/randomforest_train.py \
        --run-name randomforest_train \
        --dataset-dir /content/drive/MyDrive/colab_data/animals \
        --learning-rate 0.001 \
        --batch-size 16 \
        --num-workers 2 \
        --num-epochs 50 \
        --load-checkpoints 0 \
        --load-checkpoints-path /home \
        --save-checkpoints 1 \
        --save-checkpoints-epoch 10

elif [ "$MODEL_NAME" == "alexnet" ]; then
    echo "not implemented yet"
else
    echo "Invalid model name"
    exit 1
fi