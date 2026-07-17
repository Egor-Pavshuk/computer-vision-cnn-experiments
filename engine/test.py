import torch
from tqdm import tqdm


def test_model(model, test_loader, criterion, device=torch.device("cpu")):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    test_loop = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for images, labels in test_loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            test_loop.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy
