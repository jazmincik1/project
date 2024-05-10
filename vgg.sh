#!/bin/bash

# First, run 3 different models with 3 different learning rates for 10 epochs
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg-0.01-lr-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.01 --batch-size 32 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg-0.001-lr-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --batch-size 32 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg-0.0001-lr-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.0001 --batch-size 32 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10

# Then, for 20 epochs, one model with decay and one without decay
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg-0.001-lr-20-epochs-decay --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --batch-size 32 --num-workers 2 --num-epochs 20 --decay-lr 1 --save-checkpoints 1 --save-checkpoints-epoch 10
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg-0.001-lr-20-epochs-nodecay --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --batch-size 32 --num-workers 2 --num-epochs 20 --save-checkpoints 1 --save-checkpoints-epoch 10

# Finally, for 10 epochs, try 2 different batch sizes
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg-32-batch-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --batch-size 16 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg-64-batch-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --batch-size 64 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10

python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg11-0.001-lr-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --vgg-version '11' --batch-size 32 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg13-0.001-lr-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --vgg-version '13' --batch-size 32 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10
python scripts/vgg/vgg_fine_tune_without_plots.py --run-name vgg19-0.001-lr-10-epochs --dataset-dir /content/drive/MyDrive/colab_data/animals --learning-rate 0.001 --vgg-version '19' --batch-size 32 --num-workers 2 --num-epochs 10 --save-checkpoints 1 --save-checkpoints-epoch 10


