import sys
import os
import numpy as np

from dataset.dataset import AnimalDataset
from models.vgg import VGG

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import torch
from sklearn.model_selection import train_test_split

from src.models.randomforest import RandomForestModel
from torch.utils.data import DataLoader
from src.utils.transform import get_transform

from src.utils.log import log
from src.utils.plot_confusion_matrix import plot_confusion_matrix
from src.config.args import parser
from src.constants.constants import CLASS_NAMES



def train(model, X_train, y_train,n_est, args):
    log(f"Training Random forest with {n_est} estimators")
    loss, accuracy= model.train(X_train, y_train)
    return loss, accuracy

def test(model,X_test, y_test,n_est, args):
    log(f"Testing Random forest with {n_est} estimators")
    test_loss = 0
    all_preds = []
    all_labels = []

    predictions, accuracy, test_loss = model.test(X_test, y_test)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy}%)\n"
    )

    all_preds = predictions
    all_labels = y_test

    plot_confusion_matrix(all_labels, all_preds, f"confusion_matrix_forest_n_{n_est}", args)
    
def validate(model, X_val, y_val, n_est, args):
    log(f"Validating Random forest with {n_est} estimators")  
    test_loss = 0
    all_preds = []
    all_labels = []

    # Run the test method of the model which returns predictions, accuracy, and test loss
    predictions, acc, test_loss = model.test(X_val, y_val)

    # Print out the loss and accuracy
    print(f"\nValidation set: Average loss: {test_loss:.4f}, Accuracy: {acc}%)\n")
    
    # Return the validation accuracy
    return acc





def main(args):
    full_dataset = AnimalDataset(root_dir=args.dataset_dir, transform=get_transform(resize=256, crop=224))

    model, num_features = VGG(version=64, num_classes=len(CLASS_NAMES)).get_feature_extractor()

    data_loader = DataLoader(
        full_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    log("Loading dataset")
    
    
    model.eval()  # Set the model to evaluation mode
    device = args.device
    feature_dataset  = []
    labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            outputs_np = outputs.detach().cpu().numpy()
            feature_dataset.extend(outputs_np)
            labels.extend(labels)
            
    feature_dataset = np.array(feature_dataset)
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(feature_dataset, labels, test_size=0.2, random_state=42)
    
    #validation will be implemented later
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    best_val_acc = 0
    best_n_est = 0
    log("Start training")
    for n_est in [10,50,100,1000]:
        log(f"Random forest with {n_est} estimators")
        model = RandomForestModel(n_est)
        train(model, X_train, y_train,n_est, args)
        acc = validate(model,X_val, y_val,n_est, args)
        
        if acc > best_val_acc:
            best_val_acc = acc
            best_n_est = n_est
            
    test(model, X_test, y_test, best_n_est, args)


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()

    args.run_name = "train__" + args.run_name

    log("Using args:", args)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/loss/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/conf/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/misclassified/", exist_ok=True)
    main(args)
