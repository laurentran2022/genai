"""Train the Module 8 Energy Model on CIFAR10 and save the weights for the API."""
import random

import numpy as np
import torch

from helper_lib.data_loader import get_cifar10_generative_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.utils import get_device, save_model

EPOCHS = 20
MODEL_PATH = "models/ebm_cifar10.pth"

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    # The energy model works on images scaled to [-1, 1]
    train_loader = get_cifar10_generative_loader("data", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), batch_size=128)

    model = get_model("EBM")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

    trained_model = train_ebm(model, train_loader, optimizer, device=device, epochs=EPOCHS,
                              alpha=0.1, steps=60, step_size=10, noise=0.005)

    save_model(trained_model.to("cpu"), MODEL_PATH)
