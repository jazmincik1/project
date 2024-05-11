import matplotlib.pyplot as plt
import os


def plot_loss(losses, file_name, args):

    time_steps = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(time_steps, losses, label="Loss", color="red", marker="x")

    plt.title("Loss Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.grid(True)

    plt.savefig(f"results/{args.run_name}/loss/{file_name}.png")
    
def plot_acc_x_loss(train_losses, train_acc, args,val=False):
    directory = f"results/{args.run_name}/train" if not val else f"results/{args.run_name}/val"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Plotting
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color) 
    ax2.plot(train_acc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Loss and Accuracy') if not val else plt.title('Validation Loss and Accuracy')
    fig.tight_layout() 

    # Save the plot
    plt.savefig(f"{directory}/training_plot.png") if not val else plt.savefig(f"{directory}/validation_plot.png")
    plt.close()
