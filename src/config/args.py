import argparse
import src.config.config as config

parser = argparse.ArgumentParser(description="CS464 Project")

parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--device", type=str, default=config.DEVICE)
parser.add_argument("--dataset-dir", type=str, default=config.DATASET_DIR)
parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
parser.add_argument("--num-epochs", type=int, default=config.NUM_EPOCHS)
parser.add_argument("--load-checkpoints", type=int, default=config.LOAD_CHECKPOINTS)
parser.add_argument("--load-checkpoints-path", type=str, default=config.LOAD_CHECKPOINTS_PATH)
parser.add_argument("--save-checkpoints", type=int, default=config.SAVE_CHECKPOINTS)
parser.add_argument("--save-checkpoints-epoch", type=int, default=config.SAVE_CHECKPOINTS_EPOCH)