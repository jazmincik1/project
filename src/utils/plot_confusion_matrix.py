import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.constants.constants import CLASS_NAMES


def plot_confusion_matrix(y_true, y_pred, file_name, args):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)

    plt.title("Confusion Matrix")

    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    plt.savefig(f"results/{args.run_name}/conf/{file_name}.png")
