import torch
import time
from models import create_model
from configs.config import *
from datasets import data_loader
from engine import train, test
from utils import plots
from utils.results import save_results

def verify_shapes(device, train_loader, model, criterion):
    images, labels = next(iter(train_loader))
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    outputs = model(images.to(device))
    print("Outputs shape:", outputs.shape)

    loss = criterion(outputs.to(device), labels.to(device))
    print("Loss:", loss.item())


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
   
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(BATCH_SIZE)

    model = create_model(MODEL_NAME).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    history = train.train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=EPOCHS)

    training_time = time.time() - start_time

    checkpoint = torch.load(f"results/models/{MODEL_NAME}_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss, test_acc = test.test_model(model, test_loader, criterion, device)

    plots.plot_loss(
    history["train_losses"],
    history["val_losses"],
    MODEL_NAME
    )

    plots.plot_accuracy(
        history["train_accuracies"],
        history["val_accuracies"],
        MODEL_NAME
    )

    save_results(
    MODEL_NAME,
    test_acc,
    test_loss,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    training_time,
    "Adam"
)

if __name__ == "__main__":
    main()