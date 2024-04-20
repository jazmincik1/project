import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from collections import deque
from sklearn.model_selection import train_test_split

from src.models.randomforest import RandomForestModel
from src.dataset.dataset_np import DatasetNP
from src.utils.log import log
from src.utils.plot_loss import plot_loss
from src.utils.plot_confusion_matrix import plot_confusion_matrix
from src.utils.save_misclassified import save_misclassified_images
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

    plot_confusion_matrix(all_labels.numpy(), all_preds.numpy(), f"confusion_matrix_forest_n{n_est}", args)


def main(args):

    log("Loading dataset")
    full_dataset = DatasetNP(root_dir=args.dataset_dir)
    images, labels = full_dataset.get_data()

    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    #validation will be implemented later
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    #TODO
    #Rather than using pixels as features, later, different
    # feature extraction methodfs will be used. Such as pooling some of th epixels, or using a feature extreaction network first

    log("Start training")
    for n_est in [10,50,100,1000]:
        log(f"Random forest with {n_est} estimators")
        model = RandomForestModel(n_est)
        train(model, X_train, y_train,n_est, args)
        test(model,X_test, y_test,n_est, args)


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
