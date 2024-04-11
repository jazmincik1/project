import os
import torch
import torch.optim as optim
import torch.nn as nn

from collections import deque
from src.config.args import parser
from src.dataset.dataset import AnimalDataset
from src.models.alexnet import AlexNet
from src.utils.plot_confusion_matrix import plot_confusion_matrix
from src.utils.plot_loss import plot_loss
from src.utils.transform import get_transform
from src.utils.log import log
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Will delete this later, just to check if the AlexNet implementation was correct or not
def test_AlexNet_imp():

    dummy_input = torch.randn(1, 3, 227, 227)
    model = AlexNet()
    output = model(dummy_input)
    print(f'Output shape: {output.shape}')
    assert output.shape == (1, 10), "Output shape is incorrect"

def train(model, device, train_loader, optimizer, epoch, loss_fn, scaler, losses, args):

    model.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    loss_tot = 0

    for batch_idx, (data, target) in progress_bar:

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():

            output = model(data)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_tot += loss.item()
        progress_bar.set_postfix(loss=loss_tot / (batch_idx + 1))
        losses.append(loss.item())
        if batch_idx % args.plot_loss_every_n_iteration == 0 and batch_idx != 0:
            plot_loss(losses, f"loss_epoch_{epoch}_idx_{batch_idx}", args)

def test(model, device, test_loader, epoch, loss_fn, args):

    model.eval()
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch}, test")
    loss_tot = 0
    true_preds = 0
    preds = []
    labels = []

    with torch.no_grad():
        for _, (data, target) in progress_bar:

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_tot += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=False)
            true_preds += (pred == target).sum().item()
            preds.append(pred)
            labels.append(target)

    loss_tot /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {loss_tot:.4f}, Accuracy: {true_preds}/{len(test_loader.dataset)} ({100. * true_preds / len(test_loader.dataset):.0f}%)\n")
    preds = torch.cat(preds).cpu()
    labels = torch.cat(labels).cpu()
    plot_confusion_matrix(labels.numpy(), preds.numpy(), f"confusion_matrix_epoch_{epoch}", args)

def main(args):

    dataset = AnimalDataset(root_dir=args.dataset_dir, transform=get_transform(resize=256, crop=224))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = args.device
    model = AlexNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    losses = deque(maxlen=1000)
    test(model, device, test_loader, -1, loss_fn, args)

    for epoch in range(1, args.num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_fn, scaler, losses, args)
        test(model, device, test_loader, epoch, loss_fn, args)
        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.run_name}/epoch_{epoch}.pth")

    if args.save_checkpoints:
        torch.save(model.state_dict(), f"checkpoints/{args.run_name}/final.pth")

if __name__ == "__main__":
    args = parser.parse_args()

    args.run_name = "train__" + args.run_name
    args.device = torch.device(args.device)
    log("Using device:", args.device)

    log("Using args:", args)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/loss/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/conf/", exist_ok=True)
    main(args)