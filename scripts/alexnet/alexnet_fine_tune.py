import os
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

import scripts.alexnet.alexnet_train
import src.models.alexnet
from src.dataset.dataset import AnimalDataset
from src.utils.transform import get_transform
from src.utils.log import log
from src.utils.plot_loss import plot_loss
from src.utils.plot_confusion_matrix import plot_confusion_matrix
from src.utils.save_misclassified import save_misclassified_images
from src.config.args import parser
from src.constants.constants import CLASS_NAMES
from torch import autocast
from tqdm import tqdm
from torchvision import models
from torchvision.models import AlexNet_Weights
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split

def validate(model, device, validation_loader, loss_fn):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            validation_loss += loss.item()
    return validation_loss / len(validation_loader)

def train(model, device, train_loader, optimizer, epoch, loss_fn, scaler, losses, args):

    model.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    loss_tot = 0

    for batch_idx, (data, target) in progress_bar:

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):

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
    misclassified_examples = []

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_tot += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=False)
            true_preds += (pred == target).sum().item()
            preds.append(pred)
            labels.append(target)
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
    loss_tot /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {loss_tot:.4f}, Accuracy: {true_preds}/{len(test_loader.dataset)} ({100. * true_preds / len(test_loader.dataset):.0f}%)\n")
    preds = torch.cat(preds).cpu()
    labels = torch.cat(labels).cpu()
    plot_confusion_matrix(labels.numpy(), preds.numpy(), f"confusion_matrix_epoch_{epoch}", args)
    save_misclassified_images(misclassified_examples, f"results/{args.run_name}/misclassified/{epoch}/")

def main(args):

    dataset = AnimalDataset(root_dir=args.dataset_dir, transform=get_transform(resize=256, crop=224))
    train_size = int(0.7 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    device = args.device
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1) #trying pretrained model
    last_layer_in_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(last_layer_in_features, len(CLASS_NAMES))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier[6].parameters():
        param.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    scaler = GradScaler()
    losses = deque(maxlen=1000)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, args.num_epochs + 1):

        '''train(model, device, train_loader, optimizer, epoch, loss_fn, scaler, losses, args)
        test(model, device, test_loader, epoch, loss_fn, args)
        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.run_name}/epoch_{epoch}.pth")
        validation_loss = validate(model, device, validation_loader, loss_fn)
        scheduler.step(validation_loss)'''

        train(model, device, train_loader, optimizer, epoch, loss_fn, scaler, losses, args)
        test(model, device, test_loader, epoch, loss_fn, args)
        validation_loss = validate(model, device, validation_loader, loss_fn)
        scheduler.step(validation_loss)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"checkpoints/{args.run_name}/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == args.early_stop_patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.run_name}/epoch_{epoch}.pth")

    if args.save_checkpoints:
        torch.save(model.state_dict(), f"checkpoints/{args.run_name}/final.pth")

if __name__ == "__main__":

    args = parser.parse_args()
    args.run_name = "fine_tune__" + args.run_name
    args.device = torch.device(args.device)
    log("Using device:", args.device)
    log("Using args:", args)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/loss/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/conf/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/misclassified/", exist_ok=True)
    main(args)