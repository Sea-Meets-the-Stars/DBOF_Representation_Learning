import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset
import dbof.io.filesystems as filesystems
from dask.distributed import Client
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as torchtransforms


# Default cutout dataset location (v2 test data).
DEFAULT_S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
DEFAULT_BUCKET = "dbof"
DEFAULT_FOLDER = "cutouts_dataset_v2_TESTING"
DEFAULT_RUN_ID = "itest_d7af6005"
DEFAULT_DATASET_NAME = "cutout_dataset_creation.zarr"


class CutoutDataSource:
    """Access to a generated DBOF cutout dataset on S3.

    Defaults point at the v2 test dataset. Channel order is read from the
    store's ``channel_names`` attr (feature channels + XC, YC).
    """

    def __init__(self, bucket=DEFAULT_BUCKET, folder=DEFAULT_FOLDER,
                 run_id=DEFAULT_RUN_ID, dataset_name=DEFAULT_DATASET_NAME,
                 s3_endpoint=DEFAULT_S3_ENDPOINT):
        self.fs, self.fs_synch = filesystems.create_s3_filesystems(s3_endpoint)
        self.reader = zarr_dataset.ZarrDatasetReader(
            bucket=bucket, folder=folder, run_id=run_id,
            dataset_name=dataset_name, fs=self.fs,
        )
        self.channel_names = self.reader.channel_names

    def full_dataset_as_dask(self):
        return self.reader.full_dataset_as_dask()

    def print_available_channels(self):
        """Print every channel in the source dataset, in stored order."""
        channels_str = "["
        print(f"{len(self.channel_names)} available channels:")
        for i, name in enumerate(self.channel_names):
            channels_str += f"'{name}',"
        channels_str = channels_str[:-1]
        channels_str += "]"
        print(channels_str)


class Cutouts(Dataset):
    def __init__(self, X, channel_names=None, transform=None):
        """
        X: array/tensor of shape [N, C, H, W]
        channel_names: names of the C channels, in order
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.channel_names = channel_names
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample


def make_dataloader(X, mean, std, channel_names=None, batch_size=64, num_workers=0, shuffle=False):
    transforms = torchtransforms.Compose([
        torchtransforms.Normalize(mean=mean, std=std)
    ])

    train_ds = Cutouts(X, channel_names=channel_names, transform=transforms)
    print(f"Channels : {train_ds.channel_names}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
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


def download_data(source=None, subset=True, subsample_per_chunk=300, num_sample_chunks=30, n_workers=8):
    client = Client(n_workers=n_workers)
    print(client)
    port = client.scheduler_info()["services"]["dashboard"]
    # For nrp link is :
    # https://jupyterhub-west.nrp-nautilus.io/hub/user-redirect/proxy/{port}/status
    print(f"nrp link url : https://jupyterhub-west.nrp-nautilus.io/hub/user-redirect/proxy/{port}/status")

    source = source or CutoutDataSource()
    images_da, ids_da, valid_mask_da = source.full_dataset_as_dask()

    # drop empty store slots (rejected cutouts / failed steps)
    valid_idx = np.flatnonzero(np.asarray(valid_mask_da))
    images_da = images_da[valid_idx]
    ids_da = ids_da[valid_idx]

    if subset:
        N = len(images_da)
        subset_idxs = chunk_aware_subsample(images_da, num_sample_chunks, subsample_per_chunk)
        images_da = images_da[subset_idxs]
        ids_da = ids_da[subset_idxs]

    images_np = images_da.compute()

    return images_np


def filter_based_on_mask(data, bad_mask):
    """Keep rows where bad_mask is False."""
    return data[~bad_mask]


def filter_invalid_cutouts(images_np, channel_names):
    n_start = images_np.shape[0]

    # Drop cutouts containing ANY sea ice (SIarea > 0 anywhere in the cutout).
    siarea = images_np[:, channel_names.index("SIarea")]      # (N, H, W)
    has_ice = (siarea > 0).any(axis=(1, 2))
    images_np = filter_based_on_mask(images_np, has_ice)
    print(f"dropped {int(has_ice.sum())} cutouts with sea ice")

    # Drop cutouts containing any NaN (e.g. residual land leakage).
    has_nan = np.isnan(images_np).any(axis=(1, 2, 3))
    images_np = filter_based_on_mask(images_np, has_nan)
    print(f"dropped {int(has_nan.sum())} cutouts with NaN")

    print(f"kept {images_np.shape[0]} / {n_start} cutouts")
    return images_np


def select_channels(images_np, source_channels, data_channels, coord_channels):
    """Reorder to [*data_channels, *coord_channels]; return (images, mean, std, order).

    Data channels are z-scored. Coord channels (e.g. XC, YC) carry positional
    info used downstream and are passed through unnormalized (mean 0, std 1).
    """
    order = list(data_channels) + list(coord_channels)
    missing = [c for c in order if c not in source_channels]
    if missing:
        raise ValueError(f"channels not in dataset: {missing}. available: {source_channels}")

    idx = [source_channels.index(c) for c in order]
    images_np = images_np[:, idx]

    n_data = len(data_channels)
    mean = images_np.mean(axis=(0, 2, 3)).astype("float32")
    std = images_np.std(axis=(0, 2, 3)).astype("float32")
    mean[n_data:] = 0.0     # leave coord channels unshifted
    std[n_data:] = 1.0      # and unscaled
    return images_np, torch.from_numpy(mean), torch.from_numpy(std), order


def get_cutout_loader(source=None, data_channels=None, coord_channels=("XC", "YC"),
                      subset=True, subsample_per_chunk=300, num_sample_chunks=30,
                      n_workers=8, batch_size=64):
    source = source or CutoutDataSource()
    if data_channels is None:
        raise ValueError("pass data_channels; see source.print_available_channels()")

    images_np = download_data(source=source, subset=subset, subsample_per_chunk=subsample_per_chunk,
                              num_sample_chunks=num_sample_chunks, n_workers=n_workers)
    images_np = filter_invalid_cutouts(images_np, source.channel_names)

    images_np, mean, std, channel_order = select_channels(
        images_np, source.channel_names, data_channels, coord_channels)

    data_loader = make_dataloader(images_np, mean, std, channel_names=channel_order,
                                  batch_size=batch_size, num_workers=0)
    return data_loader
