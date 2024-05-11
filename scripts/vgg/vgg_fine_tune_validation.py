import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from collections import deque

from src.models.vgg import VGG
from src.dataset.dataset import AnimalDataset
from src.utils.transform import get_transform
from src.utils.log import log
from src.utils.plot_loss import plot_loss, plot_acc_x_loss
from src.utils.plot_confusion_matrix import plot_confusion_matrix
from src.utils.save_misclassified import save_misclassified_images
from src.config.args import parser
from src.constants.constants import CLASS_NAMES

torch.cuda.empty_cache()

def train(model, device, train_loader, optimizer, epoch, criterion, scaler, losses, scheduler, args):
    model.train()
    total_loss = 0
    correct = 0  # Correct predictions counter
    total = 0    # Total predictions counter

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        progress_bar.set_postfix(loss=total_loss / (batch_idx + 1), accuracy=100. * correct / total)

        losses.append(loss.item())
    
        if batch_idx % args.plot_loss_every_n_iteration == 0 and batch_idx != 0:
            plot_loss(losses, f"loss_epoch_{epoch}_idx_{batch_idx}", args)
            
    if args.decay_lr and scheduler is not None:
        scheduler.step()

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100. * correct / total  # Calculate total accuracy for the epoch

    return epoch_loss, epoch_accuracy
    


def validate_model(model, device, val_loader, epoch, criterion, args):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100 * correct / total

        print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Epoch: {epoch}')

    return epoch_accuracy, epoch_loss


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

            for i in range(max(misclassified_data.size(0),100)):
                example = {
                    "data": misclassified_data[i],
                    "true_label": CLASS_NAMES[misclassified_targets[i].item()],
                    "predicted_label": CLASS_NAMES[misclassified_preds[i].item()],
                }
                misclassified_examples.append(example)

    plot_confusion_matrix(all_labels.numpy(), all_preds.numpy(), f"confusion_matrix_epoch", args)
    save_misclassified_images(misclassified_examples, f"results/{args.run_name}/misclassified/")
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

    all_preds = torch.cat(all_preds).cpu()
    all_labels = torch.cat(all_labels).cpu()


def main(args):
    full_dataset = AnimalDataset(root_dir=args.dataset_dir, transform=get_transform(resize=256, crop=224))

    # Splitting the dataset into train and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    validation_size = int(0.2 * train_size)
    train_size = train_size - validation_size
    
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True
    )

    device = args.device
    vgg_version = args.vgg_version
    lr_decay = args.decay_lr

    log(f"Loading pre-trained VGG{vgg_version} model")

    model = VGG(version=vgg_version, num_classes=len(CLASS_NAMES)).get_model()

    log(f"Loaded pre-trained VGG{vgg_version} model")

    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    #add decay to learning rate
    if lr_decay:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        scheduler = None
        
    scaler = GradScaler()
    losses = deque(maxlen=1000)
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    best_loss  = 1000000
    best_model = None
    
    for epoch in range(1, args.num_epochs + 1):
        loss, acc = train(model, device, train_loader, optimizer, epoch, criterion, scaler, losses,scheduler, args)
        train_losses.append(loss)
        train_acc.append(acc)
        loss,acc = validate_model(model, device, validation_loader, epoch, criterion, args)
        val_losses.append(loss)
        val_acc.append(acc)
        
        if loss < best_loss:
            best_loss = loss
            best_model = model
        

    if args.save_checkpoints:
        torch.save(best_model.state_dict(), f"results/{args.run_name}/checkpoints/best.pth")
        torch.save(model.state_dict(), f"results/{args.run_name}/checkpoints/final.pth")
        
        
    test(best_model, device, test_loader, epoch, criterion, args)

    plot_acc_x_loss(train_losses,train_acc,args)
    plot_acc_x_loss(val_losses,val_acc,args,args,val=True)
      

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()

    args.run_name = "fine_tune__" + args.run_name
    args.device = torch.device(args.device)
    log("Using device:", args.device)

    log("Using args:", args)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/loss/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/conf/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/misclassified/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/train",exist_ok=True)
    main(args)
