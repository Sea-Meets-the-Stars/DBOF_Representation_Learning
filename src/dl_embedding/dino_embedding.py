"""Pretrained-DINO patch embedding step for the NEMI clustering pipeline.

Replaces CutoutDataset.get_patches (itself a stand-in for a real transformer)
with a frozen DINOv2 ViT.  Each cutout is reduced to 3 pseudo-RGB channels
(PCA over the feature channels), resized to the ViT input, and passed through
the backbone; the per-patch tokens (x_norm_patchtokens) become the units NEMI
clusters -- one embedding vector per ViT patch, row-major within each cutout,
cutouts concatenated in dataset order (matching get_patches' patch ordering so
the existing visualization functions work unchanged).
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# ImageNet stats DINOv2 was trained with.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_DINOV2_PATCH = 14   # ViT-S/14 token size


def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DinoPatchEmbedder:
    """Frozen DINOv2 encoder producing per-ViT-patch embeddings for clustering.

    Args:
        model_name: torch.hub DINOv2 entry (default ``dinov2_vits14``, 384-dim).
        device: torch device; defaults to cuda/mps/cpu autodetect.
    """

    def __init__(self, model_name="dinov2_vits14", device=None):
        self.device = device or _auto_device()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.embed_dim = self.model.embed_dim
        self.pca = None
        self._pca_lo = None
        self._pca_hi = None

    def fit_pca(self, images, n_components=3, sample_pixels=200_000, seed=0):
        """Fit the channels->3 PCA on a pixel sample from ``images`` (N,C,H,W).

        Records per-component 1-99 percentiles used to scale the projected
        pseudo-RGB into [0,1] before ImageNet normalization.
        """
        N, C, H, W = images.shape
        rng = np.random.default_rng(seed)
        n = min(sample_pixels, N * H * W)
        ni, hi, wi = (rng.integers(0, N, n), rng.integers(0, H, n), rng.integers(0, W, n))
        px = images[ni, :, hi, wi]                         # (n, C)
        self.pca = PCA(n_components=n_components).fit(px)
        proj = self.pca.transform(px)                      # (n, 3)
        self._pca_lo = np.percentile(proj, 1, axis=0)
        self._pca_hi = np.percentile(proj, 99, axis=0)
        return self

    def _to_rgb(self, images):
        """(N,C,H,W) -> (N,3,H,W) float32 pseudo-RGB scaled to ~[0,1] via PCA."""
        N, C, H, W = images.shape
        flat = np.moveaxis(images, 1, -1).reshape(-1, C)   # (N*H*W, C)
        proj = np.moveaxis(self.pca.transform(flat).reshape(N, H, W, 3), -1, 1)
        lo, hi = self._pca_lo.reshape(1, 3, 1, 1), self._pca_hi.reshape(1, 3, 1, 1)
        return np.clip((proj - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype("float32")

    @torch.no_grad()
    def embed(self, images, grid, batch_size=64):
        """Per-ViT-patch embeddings for cutouts ``images`` (N,C,H,W).

        Each cutout is projected to pseudo-RGB, resized to (grid*14, grid*14),
        and passed through DINOv2; the grid*grid patch tokens are returned
        row-major.  Requires fit_pca first.

        Returns:
            (N*grid*grid, embed_dim) float32; patch order matches
            CutoutDataset.get_patches (row-major within cutout, cutouts in order).
        """
        if self.pca is None:
            raise RuntimeError("call fit_pca before embed")
        rgb = self._to_rgb(images)
        size = grid * _DINOV2_PATCH
        mean, std = _IMAGENET_MEAN.to(self.device), _IMAGENET_STD.to(self.device)

        out = []
        for i in range(0, len(rgb), batch_size):
            x = torch.from_numpy(rgb[i:i + batch_size]).to(self.device)
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # (b, grid*grid, D)
            out.append(tokens.reshape(-1, tokens.shape[-1]).cpu())
        return torch.cat(out).numpy()
