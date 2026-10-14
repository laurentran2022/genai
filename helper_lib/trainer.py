import torch


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    model.train()  # Enable training-mode behavior (dropout, batchnorm statistics)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * running_correct / running_total
        print(f"Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

    return model
