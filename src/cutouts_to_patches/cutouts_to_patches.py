import torch
from einops import rearrange
from torch.utils.data import DataLoader

def images_to_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Cut a batch of images into NxN patches.

    Args:
        images:     Tensor of shape (B, C, H, W)
        patch_size: Size of each square patch (N)

    Returns:
        Tensor of shape (B, num_patches, C, N, N)
        where num_patches = (H/N) * (W/N)
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0, f"Height {H} must be divisible by patch_size {patch_size}"
    assert W % patch_size == 0, f"Width {W} must be divisible by patch_size {patch_size}"

    patches = rearrange(
        images,
        'b c (h p1) (w p2) -> b (h w) c p1 p2',
        p1=patch_size,
        p2=patch_size
    )
    return patches


class PatchedDataLoader:
    def __init__(self, dataloader: DataLoader, patch_size: int):
        self.dataloader = dataloader
        self.patch_size = patch_size

    def __iter__(self):
        for batch in self.dataloader:
            yield images_to_patches(batch, self.patch_size)

    def __len__(self):
        return len(self.dataloader)