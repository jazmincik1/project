import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from collections import deque

from src.models.resnet18 import ResNet18
from src.dataset.dataset import AnimalDataset
from src.utils.transform import get_transform
from src.utils.log import log
from utils.plot_loss import plot_loss
from utils.plot_confusion_matrix import plot_confusion_matrix
from src.config.args import parser


def train(model, device, train_loader, optimizer, epoch, loss_fn, losses, args):
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

        losses.append(loss.item())

        if batch_idx % args.plot_loss_every_n_iteration == 0 and batch_idx != 0:
            plot_loss(losses, f"loss_epoch_{epoch}_idx_{batch_idx}", args)


def test(model, device, test_loader, epoch, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0

    correct = []
    pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()  # total number of correct predictions

            pred.append(pred)
            correct.append(target)

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

    plot_confusion_matrix(correct, pred, f"confusion_matrix_epoch_{epoch}", args)


def main(args):
    full_dataset = AnimalDataset(root_dir=args.dataset_dir, transform=get_transform(resize=256, crop=224))

    # Splitting the dataset into train and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = args.device

    model = ResNet18().to(device)
    loss_fn = nn.CrossEntropyLoss()  # this was written in the original paper
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    losses = deque(maxlen=1000)

    for epoch in range(1, args.num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_fn, losses, args)
        test(model, device, test_loader, epoch, loss_fn)

        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.run_name}/epoch_{epoch}.pth")

    if args.save_checkpoints:
        torch.save(model.state_dict(), f"checkpoints/{args.run_name}/final.pth")


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()
    log("Using args:", args)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/{args.run_name}", exist_ok=True)
    main(args)
