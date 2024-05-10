import matplotlib.pyplot as plt


def plot_accuracy(accuracies, file_name, args):

    time_steps = range(1, len(accuracies) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(time_steps, accuracies, label="Accuracy", color="red", marker="x")

    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.grid(True)

    plt.savefig(f"results/{args.run_name}/accuracy/{file_name}.png")
