import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "/"
BATCH_SIZE = 32
LEARNING_RATE = 2e-4

NUM_WORKERS = 4

NUM_EPOCHS = 10
LOAD_CHECKPOINTS = True
SAVE_CHECKPOINTS = True
LOAD_CHECKPOINTS_PATH = "checkpoints"
SAVE_CHECKPOINTS_EPOCH = 1
