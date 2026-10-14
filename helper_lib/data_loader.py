import torch
from torchvision import datasets, transforms

IMAGE_SIZE = 64

# Per-channel mean/std of the CIFAR10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transform():
    """Preprocessing shared by training and inference: resize to 64x64, convert, normalize."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_data_loader(data_dir, batch_size=32, train=True):
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=get_transform()
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=2
    )
    return loader
