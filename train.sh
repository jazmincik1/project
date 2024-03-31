#!/bin/bash

MODEL_NAME="$1"

if [ "$MODEL_NAME" == "resnet18" ]; then
    python scripts/resnet18_train.py \
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
elif [ "$MODEL_NAME" == "alexnet" ]; then
    echo "not implemented yet"
elif [ "$MODEL_NAME" == "lenet" ]; then
    echo "not implemented yet"
else
    echo "Invalid model name"
    exit 1
fi