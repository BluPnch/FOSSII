import torch
from torch import nn


LATENT_DIM = 32


class Encoder(nn.Module):
    """Map a corrupted image to the parameters of a latent Gaussian."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.posterior = nn.Linear(32 * 7 * 7, 2 * LATENT_DIM)

    def forward(self, image):
        return self.posterior(self.features(image)).chunk(2, dim=1)


class Generator(nn.Module):
    """Shared VAE decoder and GAN generator. Returns pixel logits."""

    def __init__(self):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(LATENT_DIM, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, latent):
        return self.decode(latent)


class Discriminator(nn.Module):
    """Score whether an image comes from the clean MNIST distribution."""

    def __init__(self):
        super().__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 1),
        )

    def forward(self, image):
        return self.classify(image).squeeze(1)


def reparameterize(mean, log_variance):
    return mean + torch.randn_like(mean) * (0.5 * log_variance).exp()
