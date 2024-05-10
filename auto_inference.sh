#!/bin/bash

DATASET_DIR="/content/drive/MyDrive/colab_data/animals"
BASE_CHECKPOINT_DIR="/content/drive/MyDrive/results/checkpoints"

for checkpoint in $BASE_CHECKPOINT_DIR/*.pth; do
    filename=$(basename -- "$checkpoint")
    run_name="${filename%.*}"

    python vgg_inference.py --run-name $run_name --dataset-dir $DATASET_DIR --load-checkpoints-path $checkpoint --device cuda --batch-size 32 --num-workers 2

done
