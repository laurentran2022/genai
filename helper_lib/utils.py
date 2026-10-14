import os

import torch


def get_device():
    """Pick the best available device: Apple GPU (mps), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
