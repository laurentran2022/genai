import random

import numpy as np
import torch


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    model.train()  # Enable training-mode behavior (dropout, batchnorm statistics)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * running_correct / running_total
        print(f"Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

    return model


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """Train a GAN with the standard (BCE) adversarial loss.

    model: a GAN with .generator, .discriminator and .z_dim
    criterion: nn.BCELoss() (the discriminator outputs probabilities)
    optimizer: tuple of (generator_optimizer, discriminator_optimizer)
    """
    opt_gen, opt_disc = optimizer
    generator, discriminator = model.generator, model.discriminator
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_d_loss, running_g_loss = 0.0, 0.0

        for real, _ in data_loader:  # Class labels are not used
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train discriminator: push D(real) -> 1 and D(G(z)) -> 0
            noise = torch.randn(batch_size, model.z_dim, device=device)
            fake = generator(noise)
            d_loss = (criterion(discriminator(real), real_labels)
                      + criterion(discriminator(fake.detach()), fake_labels))
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # Train generator: fool the discriminator into D(G(z)) -> 1
            g_loss = criterion(discriminator(fake), real_labels)
            opt_gen.zero_grad()
            g_loss.backward()
            opt_gen.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}: D loss={running_d_loss / len(data_loader):.4f}, "
              f"G loss={running_g_loss / len(data_loader):.4f}")

    return model


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """Train a DiffusionModel to predict the noise added to normalized images.

    criterion: the noise-prediction loss, e.g. nn.L1Loss()
    """
    model.to(device)
    model.train()
    step = 0

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in data_loader:  # Class labels are not used
            images = images.to(device)
            noises = torch.randn_like(images)
            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=device)
            noise_rates, signal_rates = model.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            loss = criterion(model(noisy_images, noise_rates ** 2), noises)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            model.update_ema(step)
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}: Loss={running_loss / len(data_loader):.4f}", flush=True)

    return model


class SampleBuffer:
    """Replay buffer of Langevin samples, so sampling does not restart from noise at every step."""

    def __init__(self, model, device, batch_size=128, max_size=8192):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_size = max_size
        self.examples = [self.random_images(1) for _ in range(batch_size)]

    def random_images(self, count):
        return torch.rand((count, 3, 32, 32), device=self.device) * 2 - 1

    def sample(self, steps, step_size, noise):
        # Restart around 5% of the chains from noise and continue the rest from the buffer
        n_new = np.random.binomial(self.batch_size, 0.05)
        old_images = torch.cat(random.choices(self.examples, k=self.batch_size - n_new), dim=0)
        images = torch.cat([self.random_images(n_new), old_images], dim=0)

        new_images = self.model.langevin(images, steps, step_size, noise)

        self.examples = (list(torch.split(new_images, 1, dim=0)) + self.examples)[:self.max_size]
        return new_images


def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10, alpha=0.1, steps=60, step_size=10, noise=0.005):
    """Train an EnergyModel with contrastive divergence on images in [-1, 1].

    alpha: weight of the energy magnitude regularizer
    steps, step_size, noise: Langevin dynamics settings for the negative samples
    """
    model.to(device)
    model.train()
    buffer = SampleBuffer(model, device, batch_size=data_loader.batch_size)

    for epoch in range(epochs):
        running_loss, running_real, running_fake = 0.0, 0.0, 0.0

        for real, _ in data_loader:  # Class labels are not used
            real = real.to(device)
            real = (real + torch.randn_like(real) * noise).clamp(-1.0, 1.0)
            fake = buffer.sample(steps, step_size, noise)

            real_out = model(real)
            fake_out = model(fake)

            # Lower the energy of real images and raise the energy of generated ones
            cdiv_loss = real_out.mean() - fake_out.mean()
            reg_loss = alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"EBM loss became {loss.item()} in epoch {epoch + 1}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()
            running_real += real_out.mean().item()
            running_fake += fake_out.mean().item()

        batches = len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}: Loss={running_loss / batches:.4f}, "
              f"E(real)={running_real / batches:.4f}, E(fake)={running_fake / batches:.4f}", flush=True)

    return model
