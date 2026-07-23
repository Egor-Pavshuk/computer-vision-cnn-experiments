import torch
from tqdm import tqdm
from pathlib import Path
from configs import MODEL_NAME, CHECKPOINT_PATH, MODEL_STATE_NAME


def train_model(model, train_loader, val_loader, criterion, optimizer, device=torch.device("cpu"), num_epochs=10):
    save_dir = Path(CHECKPOINT_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):

        model.train()

        epoch_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)

        model.eval()

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, epoch, num_epochs)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                MODEL_STATE_NAME: model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            # Save the best model
            torch.save(checkpoint, save_dir / f"{MODEL_NAME}_model.pth")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    correct_predictions = 0
    total_predictions = 0
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    train_acc = correct_predictions / total_predictions if total_predictions > 0 else 0

    return epoch_loss, train_acc

def evaluate_model(model, val_loader, criterion, device, epoch, num_epochs):
    val_loss = 0.0
    eval_loop = tqdm(val_loader, desc=f"Evaluating Epoch {epoch+1}/{num_epochs}")
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in eval_loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            eval_loop.set_postfix(loss=loss.item())

    val_loss /= len(val_loader)
    val_acc = correct_predictions / total_predictions if total_predictions > 0 else 0

    return val_loss, val_acc
