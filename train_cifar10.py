"""Train the Assignment 2 CNN on CIFAR10 and save the weights for the API."""
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, save_model

EPOCHS = 10
MODEL_PATH = "models/cnn_cifar10.pth"

if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_data_loader("data", batch_size=64, train=True)
    test_loader = get_data_loader("data", batch_size=64, train=False)

    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)
    test_loss, test_accuracy = evaluate_model(trained_model, test_loader, criterion, device=device)
    print(f"Final test accuracy: {test_accuracy:.2f}%")

    save_model(trained_model.to("cpu"), MODEL_PATH)
