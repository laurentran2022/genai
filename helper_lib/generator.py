import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def _plot_grid(images, num_samples):
    grid = make_grid(images, nrow=min(num_samples, 10), padding=2)
    plt.figure(figsize=(min(num_samples, 10), (num_samples + 9) // 10))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()


def generate_samples(model, device, num_samples=10, diffusion_steps=100):
    """Sample images from a GAN or DiffusionModel and plot them on a grid.

    diffusion_steps is only used by a DiffusionModel.
    """
    model.to(device)
    model.eval()  # Use running batchnorm statistics

    if hasattr(model, "ema_network"):
        images = model.generate(num_samples, diffusion_steps).cpu()
    else:
        with torch.no_grad():
            noise = torch.randn(num_samples, model.z_dim, device=device)
            images = model.generator(noise).cpu() * 0.5 + 0.5  # Undo the [-1, 1] normalization

    _plot_grid(images, num_samples)


def generate_ebm_samples(model, device, num_samples=10, steps=256):
    """Sample images from an EnergyModel with Langevin dynamics and plot them on a grid."""
    model.to(device)
    model.eval()
    _plot_grid(model.generate(num_samples, steps).cpu(), num_samples)
