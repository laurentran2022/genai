import torch
import torch.nn as nn
import torch.nn.functional as F


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


MODELS = {
    "FCNN": FCNN,
    "CNN": CNN,
    "EnhancedCNN": EnhancedCNN,
    "GAN": GAN,
}


def get_model(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODELS)}")
    model = MODELS[model_name]()
    return model
