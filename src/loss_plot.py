import matplotlib.pyplot as plt

def plot_loss(loss_history):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (house, losses) in zip(axes, loss_history.items()):
        ax.plot(losses)
        ax.set_title(house)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("loss.png")
    plt.close()
