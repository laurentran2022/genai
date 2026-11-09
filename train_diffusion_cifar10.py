"""Train the Module 8 Diffusion Model on CIFAR10 and save the weights for the API."""
import torch
import torch.nn as nn

from helper_lib.data_loader import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_generative_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import get_device, save_model

EPOCHS = 20
MODEL_PATH = "models/diffusion_cifar10.pth"

if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    train_loader = get_cifar10_generative_loader("data", CIFAR10_MEAN, CIFAR10_STD, batch_size=128)

    model = get_model("Diffusion")
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()

    trained_model = train_diffusion(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)

    save_model(trained_model.to("cpu"), MODEL_PATH)
