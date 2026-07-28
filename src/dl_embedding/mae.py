"""Multi-channel ViT-MAE embedding step for the clustering pipeline.

A plain masked autoencoder trained on whole cutouts (N, C, H, W).  The patch
embed is a C-channel conv, so all feature channels are ingested natively (no
PCA).  With patch_size=8 the encoder tokens form an (H/8 x W/8) grid -- one
embedding per 8x8 region, i.e. patch-level, matching the DINO/VAE granularity.
Encoder tokens are flattened row-major within cutout (cutouts in dataset order),
so the existing visualization functions work unchanged with patch_size=8.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset


def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encoder(dim, depth, heads, mlp_ratio):
    layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio),
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth)


class PatchMAE(nn.Module):
    def __init__(self, img_size, in_chans, patch_size=8, embed_dim=192, depth=6,
                 num_heads=6, decoder_dim=96, decoder_depth=2, decoder_heads=3, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size, self.in_chans = patch_size, in_chans
        self.grid = img_size // patch_size
        num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.encoder = _encoder(embed_dim, depth, num_heads, mlp_ratio)
        self.enc_norm = nn.LayerNorm(embed_dim)

        self.dec_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.decoder = _encoder(decoder_dim, decoder_depth, decoder_heads, mlp_ratio)
        self.dec_norm = nn.LayerNorm(decoder_dim)
        self.dec_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

        for p in (self.pos_embed, self.decoder_pos_embed, self.mask_token):
            nn.init.trunc_normal_(p, std=0.02)

    def patchify(self, imgs):
        p = self.patch_size
        return rearrange(imgs, "b c (gh p1) (gw p2) -> b (gh gw) (p1 p2 c)", p1=p, p2=p)

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        ids_shuffle = torch.argsort(torch.rand(B, N, device=x.device), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs, mask_ratio):
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2) + self.pos_embed   # (B, N, D)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        return self.enc_norm(self.encoder(x)), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.dec_embed(x)
        D = x.shape[-1]
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] - x.shape[1], -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D)) + self.decoder_pos_embed
        return self.dec_pred(self.dec_norm(self.decoder(x)))                      # (B, N, p*p*C)

    def forward(self, imgs, mask_ratio):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = ((pred - self.patchify(imgs)) ** 2).mean(dim=-1)                   # (B, N)
        return (loss * mask).sum() / mask.sum()                                   # masked patches only

    def encode(self, imgs):
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2) + self.pos_embed
        return self.enc_norm(self.encoder(x))                                     # (B, N, D), no masking


class MAEEmbedder:
    """Train a multi-channel ViT-MAE on cutouts; per-patch embedding = encoder token.

    fit(train_images, val_images) trains with per-epoch train/val loss;
    embed(images) returns (N*grid*grid, embed_dim), row-major within cutout.
    """

    def __init__(self, patch_size=8, embed_dim=192, depth=6, num_heads=6,
                 decoder_dim=96, decoder_depth=2, decoder_heads=3, mask_ratio=0.75, device=None):
        self.cfg = dict(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                        decoder_dim=decoder_dim, decoder_depth=decoder_depth, decoder_heads=decoder_heads)
        self.mask_ratio = mask_ratio
        self.device = device or _auto_device()
        self.model = None

    def _loader(self, images, batch_size, shuffle):
        X = torch.from_numpy(np.asarray(images, dtype="float32"))
        return DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=shuffle)

    def _epoch_loss(self, loader):
        self.model.eval()
        tot = n = 0
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                tot += self.model(xb, self.mask_ratio).item() * len(xb)
                n += len(xb)
        return tot / n

    def fit(self, train_images, val_images=None, epochs=50, batch_size=64,
            lr=1.5e-4, weight_decay=0.05, seed=0):
        torch.manual_seed(seed)
        X = np.asarray(train_images, dtype="float32")            # (N, C, H, W)
        self.model = PatchMAE(img_size=X.shape[2], in_chans=X.shape[1], **self.cfg).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader = self._loader(X, batch_size, shuffle=True)
        val_loader = self._loader(val_images, batch_size, shuffle=False) if val_images is not None else None

        for ep in range(epochs):
            t0 = time.time()
            self.model.train()
            tot = n = 0
            for (xb,) in train_loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                loss = self.model(xb, self.mask_ratio)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
                n += len(xb)
            msg = f"epoch {ep + 1}/{epochs} | train {tot / n:.4f}"
            if val_loader is not None:
                msg += f" | val {self._epoch_loss(val_loader):.4f}"
            print(msg + f" | {time.time() - t0:.1f}s")
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
            out.append(rearrange(self.model.encode(xb), "b n d -> (b n) d").cpu())
        return torch.cat(out).numpy()
