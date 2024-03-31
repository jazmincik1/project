import matplotlib.pyplot as plt


def plot_loss(losses, file_name, args):

    time_steps = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(time_steps, losses, label="Generator Loss", color="red", marker="x")

    plt.title("Loss Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.grid(True)

    plt.savefig(f"results/{args.run_name}/loss/{file_name}.png")
