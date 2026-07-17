from pathlib import Path
import csv


RESULTS_DIR = Path("results") / "csv"
RESULTS_DIR.mkdir(exist_ok=True)


def save_results(
    model_name,
    accuracy,
    loss,
    epochs,
    batch_size,
    learning_rate,
    training_time,
    optimizer
):
    csv_path = RESULTS_DIR / f"{model_name}_results.csv"

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Model",
                "Accuracy",
                "Loss",
                "Epochs",
                "Batch Size",
                "Learning Rate",
                "Training Time",
                "Optimizer"
            ])

        writer.writerow([
            model_name,
            round(accuracy, 4),
            round(loss, 4),
            epochs,
            batch_size,
            learning_rate,
            round(training_time, 2),
            optimizer
        ])