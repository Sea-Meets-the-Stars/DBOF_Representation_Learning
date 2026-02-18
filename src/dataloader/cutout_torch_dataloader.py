from dask.distributed import Client
import torch
from torch.utils.data import Dataset, DataLoader
from dask.distributed import progress
from dask.diagnostics import ProgressBar

class DBOFCutoutInMemoryDataset(Dataset):

    def __init__(self, reader, client, transform=None, subset=None):
        self.reader = reader
        self.transform = transform
        self.subset = subset

        images_da, ids_da, valid_mask_da = reader.full_dataset_as_dask()

        if self.subset is not None:
            print("Loading Images into memory...")
            with ProgressBar():
                images_np = images_da[:subset].compute()

            print("Loading ids into memory...")
            with ProgressBar():
                ids_np = ids_da[:subset].compute()


        else:
            print("Loading Images into memory...")
            with ProgressBar():
                images_np = images_da.compute()

            print("Loading ids into memory...")
            with ProgressBar():
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


def make_dbof_cutout_dataloader(reader, dask_client, batch_size=64, transform=None, subset=None):
    dataset = DBOFCutoutInMemoryDataset(reader, dask_client, transform=transform, subset=subset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # with dask loading we cannot support multiple workers
        pin_memory=True,
        persistent_workers=False,
    )
    return loader