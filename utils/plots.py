import matplotlib.pyplot as plt
from pathlib import Path
from configs import *

PLOTS_DIR = Path(RESULTS_PATH) / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def plot_metric(train_values, val_values, ylabel, filename):
    plt.figure(figsize=(8, 5))

    plt.plot(train_values, label=f"Train {ylabel}")
    plt.plot(val_values, label=f"Validation {ylabel}")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.legend()
    plt.grid(True)

    plt.savefig(PLOTS_DIR / filename)
    plt.close()


def plot_loss(train_losses, val_losses, model_name):
    plot_metric(
        train_losses,
        val_losses,
        ylabel="Loss",
        filename=f"{model_name}_loss.png"
    )


def plot_accuracy(train_accs, val_accs, model_name):
    plot_metric(
        train_accs,
        val_accs,
        ylabel="Accuracy",
        filename=f"{model_name}_accuracy.png"
    )