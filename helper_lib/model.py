import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from helper_lib.data_loader import CIFAR10_MEAN, CIFAR10_STD


class FCNN(nn.Module):
    """Fully connected network for 32x32x3 CIFAR10 images."""

    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    """Assignment 2 CNN for 64x64x3 RGB input and 10 output classes."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 64x64x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 32x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves the spatial size
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = x.view(x.size(0), -1)  # Flatten to 32*16*16 = 8192
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EnhancedCNN(nn.Module):
    """Deeper CNN with BatchNorm and Dropout, from the Module 3 notebook (32x32x3 input)."""

    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class Generator(nn.Module):
    """Assignment 3 GAN generator: noise vector (N, 100) -> 1x28x28 image in [-1, 1]."""

    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # 14x14 -> 28x28

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 7, 7)  # Reshape to 128x7x7
        x = F.relu(self.bn1(self.deconv1(x)))
        return torch.tanh(self.deconv2(x))


class Discriminator(nn.Module):
    """Assignment 3 GAN discriminator: 1x28x28 image -> probability that the image is real."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14x14 -> 7x7
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = x.view(x.size(0), -1)  # Flatten to 128*7*7 = 6272
        return torch.sigmoid(self.fc(x))


class GAN(nn.Module):
    """Container holding the generator and discriminator so they can be built, saved and loaded together."""

    def __init__(self, z_dim=100):
        super(GAN, self).__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """Module 8 energy network: 3x32x32 CIFAR10 image in [-1, 1] -> scalar energy (low = realistic)."""

    def __init__(self):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 4x4 -> 2x2
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten to 64*2*2 = 256
        x = swish(self.fc1(x))
        return self.fc2(x)

    def langevin(self, images, steps, step_size, noise_std):
        """Run Langevin dynamics: noisy gradient descent on the energy with respect to the pixels."""
        for _ in range(steps):
            with torch.no_grad():
                images = (images + torch.randn_like(images) * noise_std).clamp(-1.0, 1.0)
            images.requires_grad_(True)
            energy = self(images)
            grads, = torch.autograd.grad(energy, images, grad_outputs=torch.ones_like(energy))
            with torch.no_grad():
                grads = grads.clamp(-0.03, 0.03)  # Gradient clipping stabilizes the sampling
                images = (images - step_size * grads).clamp(-1.0, 1.0)
        return images.detach()

    def generate(self, num_images, steps=256, step_size=10.0, noise_std=0.01):
        """Start from uniform noise, run Langevin dynamics, and return images in [0, 1]."""
        device = next(self.parameters()).device
        images = torch.rand((num_images, 3, 32, 32), device=device) * 2 - 1
        return self.langevin(images, steps, step_size, noise_std) * 0.5 + 0.5


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Map diffusion times in [0, 1] to (noise_rates, signal_rates)."""
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    return torch.sin(diffusion_angles), torch.cos(diffusion_angles)


class SinusoidalEmbedding(nn.Module):
    """Encode a noise variance of shape (B, 1, 1, 1) as (B, 1, 1, 2 * num_frequencies)."""

    def __init__(self, num_frequencies=16):
        super(SinusoidalEmbedding, self).__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        x = x.expand(-1, 1, 1, self.num_frequencies)
        return torch.cat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(swish(self.conv1(x))) + self.proj(x)


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super(DownBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super(UpBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = block(torch.cat([x, skips.pop()], dim=1))
        return x


class UNet(nn.Module):
    """Module 8 U-Net: predicts the noise in an image given the noise variance."""

    def __init__(self, image_size=32, num_channels=3):
        super(UNet, self).__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)

        self.down1 = DownBlock(32, block_depth=2, in_channels=64)
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances).permute(0, 3, 1, 2)  # Shape: (B, 32, 1, 1)
        noise_emb = F.interpolate(noise_emb, size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    """U-Net plus its EMA copy, the diffusion schedule, and the CIFAR10 normalization."""

    def __init__(self, image_size=32, num_channels=3, ema_decay=0.999):
        super(DiffusionModel, self).__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.ema_decay = ema_decay
        self.schedule_fn = offset_cosine_diffusion_schedule
        self.network = UNet(image_size, num_channels)
        self.ema_network = copy.deepcopy(self.network)
        for parameter in self.ema_network.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("normalizer_mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("normalizer_std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, noisy_images, noise_variances):
        return self.network(noisy_images, noise_variances)

    @torch.no_grad()
    def update_ema(self, step):
        # Warm-up: without it the average stays close to the random initialization early on
        decay = min(self.ema_decay, (1 + step) / (10 + step))
        for ema_parameter, parameter in zip(self.ema_network.parameters(), self.network.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)

    def denormalize(self, images):
        return torch.clamp(images * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    @torch.no_grad()
    def generate(self, num_images, diffusion_steps=100):
        """Run deterministic reverse diffusion from Gaussian noise and return images in [0, 1]."""
        self.ema_network.eval()
        device = self.normalizer_mean.device
        current_images = torch.randn((num_images, self.num_channels, self.image_size, self.image_size), device=device)
        step_size = 1.0 / diffusion_steps
        for step in range(diffusion_steps):
            t = torch.ones((num_images, 1, 1, 1), device=device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises = self.ema_network(current_images, noise_rates ** 2)
            pred_images = (current_images - noise_rates * pred_noises) / signal_rates
            next_noise_rates, next_signal_rates = self.schedule_fn(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return self.denormalize(pred_images)


MODELS = {
    "FCNN": FCNN,
    "CNN": CNN,
    "EnhancedCNN": EnhancedCNN,
    "GAN": GAN,
    "EBM": EnergyModel,
    "Diffusion": DiffusionModel,
}


def get_model(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODELS)}")
    model = MODELS[model_name]()
    return model
