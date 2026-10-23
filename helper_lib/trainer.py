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
