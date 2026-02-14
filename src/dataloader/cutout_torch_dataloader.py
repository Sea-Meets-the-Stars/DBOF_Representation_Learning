from dask.distributed import Client
import torch
from torch.utils.data import Dataset, DataLoader

class DBOFCutoutInMemoryDataset(Dataset):

    def __init__(self, reader, subset= None, transform=None):
        client = Client()

        self.reader = reader
        self.transform = transform

        images_da, ids_da, valid_mask_da = reader.full_dataset_as_dask()

        if subset is not None:
            images_da = images_da[:subset]
            ids_da = ids_da[:subset]

        # subset for now
        print(f"Loading Images into memory {images_da.shape} ...")
        images_np = images_da.compute()
        print(f"Loading ids into memory {ids_da.shape} ...")
        ids_np = ids_da.compute()
        mask = (ids_np != b"")

        self.images = images_np[mask]
        self.ids = ids_np[mask]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to torch tensor
        x = torch.as_tensor(self.images[idx])

        if x.dtype != torch.float32:
            x = x.float()

        if self.transform is not None:
            x = self.transform(x)

        return x

def make_dbof_cutout_dataloader(reader, subset=None, batch_size=64, num_workers=0, transform=None):
    dataset = DBOFCutoutInMemoryDataset(reader, subset=subset, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # works (map-style)
        drop_last=True,  # common for SSL/DINO
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader