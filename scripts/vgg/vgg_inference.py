import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import deque

from src.models.vgg import VGG
from src.dataset.dataset import AnimalDataset
from src.utils.transform import get_transform
from src.utils.log import log
from src.utils.plot_confusion_matrix import plot_confusion_matrix
from src.config.args import parser
from src.constants.constants import CLASS_NAMES

def test(model, device, test_loader, criterion, args):
    model.eval()
    test_loss = 0
    correct = 0

    all_preds = []
    all_labels = []
    misclassified_examples = []

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test")

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += (pred == target).sum().item()

            all_preds.append(pred)
            all_labels.append(target)

            # Identify misclassified examples
            misclassified_indices = pred != target
            misclassified_data = data[misclassified_indices]
            misclassified_targets = target[misclassified_indices]
            misclassified_preds = pred[misclassified_indices]

            for i in range(misclassified_data.size(0)):
                example = {
                    "data": misclassified_data[i],
                    "true_label": CLASS_NAMES[misclassified_targets[i].item()],
                    "predicted_label": CLASS_NAMES[misclassified_preds[i].item()],
                }
                misclassified_examples.append(example)

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

    all_preds = torch.cat(all_preds).cpu()
    all_labels = torch.cat(all_labels).cpu()

    plot_confusion_matrix(all_labels.numpy(), all_preds.numpy(), f"confusion_matrix_test", args)
    
def inference(path):
    model = VGG(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(path))
    
def main():
    full_dataset = AnimalDataset(root_dir=args.dataset_dir, transform=get_transform(resize=256, crop=224))

    # Splitting the dataset into train and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True
    )
    
    model = inference(args.load_checkpoints_path)
    model = model.to(args.device)
    
    criterion = nn.CrossEntropyLoss()
    
    test(model, args.device, test_loader, criterion, args)

    
    
if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()

    args.run_name = "fine_tune__" + args.run_name
    args.device = torch.device(args.device)
    log("Using device:", args.device)

    log("Using args:", args)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/test/{args.run_name}/plots/", exist_ok=True)
    os.makedirs(f"results/test/{args.run_name}/conf/", exist_ok=True)
    os.makedirs(f"results/test/{args.run_name}/misclassified/", exist_ok=True)
    main(args)