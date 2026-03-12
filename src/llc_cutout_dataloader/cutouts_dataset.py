import dbof.dataset_creation.zarr_dataset as zarr_dataset
import dbof.io.filesystems as filesystems
from dask.distributed import Client
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as torchtransforms


class Cutouts(Dataset):
    def __init__(self, X, transform=None):
        """
        X: array/tensor of shape [N, C, H, W]
        labels: array/tensor of shape [N]
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample


def make_dataloader(X, mean, std, batch_size=64, num_workers=0):
    transforms = torchtransforms.Compose([
        torchtransforms.Normalize(mean=mean, std=std)
    ])

    train_ds = Cutouts(X, transform=transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader

def chunk_aware_subsample(da, num_sample_chunks, subsample_per_chunk, chunk = 1020):
    rng = np.random.default_rng()

    n = da.shape[0]
    n_chunks = (n + chunk - 1) // chunk

    sample_chunks = rng.choice(n_chunks, size=num_sample_chunks, replace=False)

    # within each chosen chunk, pick r indices

    idx = []
    for c in sample_chunks:
        start = c * chunk
        stop = min((c + 1) * chunk, n)
        idx.append(rng.integers(start, stop, size=subsample_per_chunk))

    idx = np.sort(np.concatenate(idx))
    return idx


def download_data(subset=True, subsample_per_chunk = 300, num_sample_chunks = 30, n_workers=8):
    client = Client(n_workers=n_workers)
    print(client)
    port = client.scheduler_info()["services"]["dashboard"]
    # For nrp link is :
    # https://jupyterhub-west.nrp-nautilus.io/hub/user-redirect/proxy/{port}/status
    print(f"nrp link url : https://jupyterhub-west.nrp-nautilus.io/hub/user-redirect/proxy/{port}/status")

    bucket = "dbof"  #todo use config here
    folder = "native_grid_dbof_training_data"
    s3_endpoint = "https://s3-west.nrp-nautilus.io"
    run_id = "big_run_00"

    fs, fs_synch = filesystems.create_s3_filesystems(s3_endpoint)
    reader = zarr_dataset.ZarrDatasetReader(
        bucket=bucket,
        folder=folder,
        run_id=run_id,
        dataset_name="dataset_creation.zarr",
        fs=fs
    )

    images_da, ids_da, valid_mask_da = reader.full_dataset_as_dask()

    if subset:
        N = len(images_da)
        subset_idxs = chunk_aware_subsample(images_da, num_sample_chunks, subsample_per_chunk)
        images_da = images_da[subset_idxs]
        ids_da = ids_da[subset_idxs]

    images_np = images_da.compute()

    return images_np

# cutouts with ice, land, or nan gradients
def filter_based_on_mask(data, bool_mask):
    ice_indices = np.where(bool_mask)[0]
    N = data.shape[0]
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[ice_indices] = False
    return data[keep_mask]

def filter_invalid_cutouts(images_np, feature_channels = ['Eta', 'Salt', 'Theta', 'U', 'V', 'W', 'relative_vorticity', 'log_gradb']):

    theta = images_np[:, feature_channels.index("Theta")]  # (N, 64, 64)

    # ICE
    bad_mask = (theta <= 0).any(axis=(1, 2))
    images_np = filter_based_on_mask(images_np, bad_mask)

    # bad_mask = (theta <= 0).any(axis=(1, 2))
    # ice_indices = np.where(bad_mask)[0]
    # N = images_clean_np.shape[0]
    # keep_mask = np.ones(N, dtype=bool)
    # keep_mask[ice_indices] = False
    # images_clean_no_ice_np = images_clean_np[keep_mask]


    # Boolean mask: True if patch has any NaN
    bad_patch_mask = np.isnan(theta).reshape(theta.shape[0], -1).any(axis=1)
    images_np = filter_based_on_mask(images_np, bad_patch_mask)

    # bad_indices = np.where(bad_patch_mask)[0]
    #
    # N = images_np.shape[0]
    # keep_mask = np.ones(N, dtype=bool)
    # keep_mask[bad_indices] = False
    # images_clean_np = images_np[keep_mask]



    vort = images_np[:, feature_channels.index("relative_vorticity")]  # (N, 64, 64)
    # Boolean mask: True if patch has any NaN
    bad_patch_mask = np.isnan(vort).reshape(vort.shape[0], -1).any(axis=1)
    images_np = filter_based_on_mask(images_np, bad_patch_mask)


    # bad_indices = np.where(bad_patch_mask)[0]
    # # Filter out cutouts containing nan gradients
    # N = images_clean_np.shape[0]
    # keep_mask = np.ones(N, dtype=bool)
    # keep_mask[bad_indices] = False
    # images_clean_np = images_clean_np[keep_mask]
    return images_np


def get_cutout_loader(subset=True, subsample_per_chunk = 300, num_sample_chunks = 30, n_workers=8, batch_size=64):
    images_np = download_data(subset=subset, subsample_per_chunk=subsample_per_chunk,
                              num_sample_chunks=num_sample_chunks, n_workers=n_workers)

    images_np = filter_invalid_cutouts(images_np)

    mean = torch.tensor(images_np.mean(axis=(0, 2, 3)))
    std = torch.tensor(images_np.std(axis=(0, 2, 3)))


    data_loader = make_dataloader(images_np, mean, std, batch_size=batch_size, num_workers=0)

    return data_loader
