import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10):
    """Sample num_samples latent vectors, run the generator, and plot the images on a grid."""
    model.to(device)
    model.eval()  # Use running batchnorm statistics

    with torch.no_grad():
        noise = torch.randn(num_samples, model.z_dim, device=device)
        images = model.generator(noise).cpu()

    images = images * 0.5 + 0.5  # Undo the [-1, 1] normalization
    grid = make_grid(images, nrow=min(num_samples, 10), padding=2)
    plt.figure(figsize=(min(num_samples, 10), (num_samples + 9) // 10))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()
