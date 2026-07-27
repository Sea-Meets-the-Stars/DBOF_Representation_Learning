"""Simple convolutional VAE embedding step for the clustering pipeline (POC).

Trains on whole cutouts (N, C, H, W); the encoder bottleneck is a spatial grid
of latents (H/8 x W/8), one latent vector per 8x8 region -- so the embeddings are
patch-level, matching the DINO patch tokens.  Latent means are flattened
row-major within each cutout (cutouts in dataset order), so the existing
visualization functions work unchanged with patch_size=8.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset

_DOWNSAMPLE = 8   # three stride-2 conv blocks -> patch_size 8


def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ConvVAE(nn.Module):
    def __init__(self, in_channels, latent_dim=16, base=32):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.to_mu = nn.Conv2d(c3, latent_dim, 1)
        self.to_logvar = nn.Conv2d(c3, latent_dim, 1)
        self.from_latent = nn.Conv2d(latent_dim, c3, 1)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(c1, in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterize
        return self.dec(self.from_latent(z)), mu, logvar


def _elbo(recon, x, mu, logvar, beta):
    rec = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl


class ConvVAEEmbedder:
    """Train a light conv VAE on cutouts; per-patch embedding = latent mean grid.

    fit(images) trains on (N, C, H, W); embed(images) returns (N*grid*grid,
    latent_dim) latent means, row-major within cutout (grid = H/8 = W/8).
    """

    def __init__(self, latent_dim=16, base=32, device=None):
        self.latent_dim, self.base = latent_dim, base
        self.device = device or _auto_device()
        self.model = None

    def fit(self, images, epochs=30, batch_size=64, lr=1e-3, beta=1.0, seed=0):
        torch.manual_seed(seed)
        X = np.asarray(images, dtype="float32")            # (N, C, H, W)
        assert X.shape[2] % _DOWNSAMPLE == 0 and X.shape[3] % _DOWNSAMPLE == 0, \
            f"H, W must be divisible by {_DOWNSAMPLE}"
        self.model = ConvVAE(X.shape[1], self.latent_dim, self.base).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                            batch_size=batch_size, shuffle=True)
        self.model.train()
        for ep in range(epochs):
            tot = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                recon, mu, logvar = self.model(xb)
                loss = _elbo(recon, xb, mu, logvar, beta)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            print(f"epoch {ep + 1}/{epochs}  elbo {tot / len(X):.4f}")
        return self

    @torch.no_grad()
    def embed(self, images, batch_size=64):
        if self.model is None:
            raise RuntimeError("call fit before embed")
        X = np.asarray(images, dtype="float32")
        self.model.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            mu, _ = self.model.encode(xb)                  # (b, D, grid, grid)
            out.append(rearrange(mu, "n c h w -> (n h w) c").cpu())
        return torch.cat(out).numpy()
