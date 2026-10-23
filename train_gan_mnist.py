"""Train the Assignment 3 GAN on MNIST and save the weights for the API."""
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, save_model

EPOCHS = 20
MODEL_PATH = "models/gan_mnist.pth"

if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_mnist_loader("data", batch_size=128, train=True)

    model = get_model("GAN")
    criterion = nn.BCELoss()
    # Adam with beta1=0.5 is the standard DCGAN setting for stable adversarial training
    opt_gen = optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_disc = optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    trained_model = train_gan(model, train_loader, criterion, (opt_gen, opt_disc), device=device, epochs=EPOCHS)

    save_model(trained_model.to("cpu"), MODEL_PATH)
