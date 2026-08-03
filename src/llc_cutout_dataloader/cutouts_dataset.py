import dbof.cutout_dataset_creation.zarr_dataset as zarr_dataset
import dbof.cutout_dataset_creation.metadata as metadata_mod
import dbof.io.filesystems as filesystems
from dask.distributed import Client
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, reduce


# Default cutout dataset location (v2 test data).
DEFAULT_S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
DEFAULT_BUCKET = "dbof"
DEFAULT_FOLDER = "cutouts_dataset_v2_TESTING"
DEFAULT_RUN_ID = "itest_d7af6005"
DEFAULT_DATASET_NAME = "cutout_dataset_creation.zarr"


def _as_str(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _to_patches(images, patch_size):
    """(N, C, H, W) -> (N*ppi, C, p, p); patch order within a cutout is row-major."""
    return rearrange(images, 'n c (h p1) (w p2) -> (n h w) c p1 p2',
                     p1=patch_size, p2=patch_size)


_GRAD_PREFIX = "grad"      # grad-magnitude channels (gradb2, gradrho2, ...) are log-scaled


def _safe_log10(a):
    """log10 with a positive floor so exact-zero gradients don't produce -inf."""
    pos = a[a > 0]
    floor = float(pos.min()) if pos.size else 1.0
    return np.log10(np.maximum(a, floor))


class CutoutDataSource:
    """Access to a generated DBOF cutout dataset on S3.

    Defaults point at the v2 test dataset. Channel order is read from the
    store's ``channel_names`` attr (feature channels + XC, YC).
    """

    def __init__(self, bucket=DEFAULT_BUCKET, folder=DEFAULT_FOLDER,
                 run_id=DEFAULT_RUN_ID, dataset_name=DEFAULT_DATASET_NAME,
                 s3_endpoint=DEFAULT_S3_ENDPOINT):
        self.bucket, self.folder, self.run_id = bucket, folder, run_id
        self.fs, self.fs_synch = filesystems.create_s3_filesystems(s3_endpoint)
        self.reader = zarr_dataset.ZarrDatasetReader(
            bucket=bucket, folder=folder, run_id=run_id,
            dataset_name=dataset_name, fs=self.fs,
        )
        self.channel_names = self.reader.channel_names

    def full_dataset_as_dask(self):
        return self.reader.full_dataset_as_dask()

    def read_metadata(self):
        """Full per-cutout metadata table (parquet), indexed by image_id."""
        reader = metadata_mod.create_metadata_reader(
            self.bucket, self.folder, self.run_id, self.fs_synch)
        df = reader.read()
        return df.set_index(df["image_id"].map(_as_str))

    def print_available_channels(self):
        """Print every channel in the source dataset, in stored order."""
        print(f"{len(self.channel_names)} available channels:")
        print("[" + ",".join(f"'{n}'" for n in self.channel_names) + "]")


def chunk_aware_subsample(da, num_sample_chunks, subsample_per_chunk, chunk=1020):
    rng = np.random.default_rng()
    n = da.shape[0]
    n_chunks = (n + chunk - 1) // chunk
    sample_chunks = rng.choice(n_chunks, size=num_sample_chunks, replace=False)

    idx = []
    for c in sample_chunks:
        start = c * chunk
        stop = min((c + 1) * chunk, n)
        idx.append(rng.integers(start, stop, size=subsample_per_chunk))
    return np.sort(np.concatenate(idx))


def _download(source, subset, subsample_per_chunk, num_sample_chunks, n_workers):
    client = Client(n_workers=n_workers)
    print(client)
    port = client.scheduler_info()["services"]["dashboard"]
    print(f"nrp link url : https://jupyterhub-west.nrp-nautilus.io/hub/user-redirect/proxy/{port}/status")

    images_da, ids_da, valid_mask_da = source.full_dataset_as_dask()

    # drop empty store slots (rejected cutouts / failed steps)
    valid_idx = np.flatnonzero(np.asarray(valid_mask_da))
    images_da, ids_da = images_da[valid_idx], ids_da[valid_idx]

    if subset:
        subset_idxs = chunk_aware_subsample(images_da, num_sample_chunks, subsample_per_chunk)
        images_da, ids_da = images_da[subset_idxs], ids_da[subset_idxs]

    return images_da.compute(), np.asarray(ids_da.compute())   # aligned


def _filter_invalid(images_np, ids_np, channel_names):
    n_start = images_np.shape[0]
    siarea = images_np[:, channel_names.index("SIarea")]      # (N, H, W)
    has_ice = (siarea > 0).any(axis=(1, 2))
    has_nan = np.isnan(images_np).any(axis=(1, 2, 3))
    keep = ~(has_ice | has_nan)
    print(f"dropped {int(has_ice.sum())} ice, {int(has_nan.sum())} NaN; "
          f"kept {int(keep.sum())} / {n_start}")
    return images_np[keep], ids_np[keep]


def _split_channels(images_np, source_channels, data_channels, coord_channels):
    """Split cutouts into RAW feature array + RAW coord array, and compute the
    per-feature z-score stats.  Features and coords stay separate so coordinates
    can never leak into the training/clustering data.

    X keeps exactly ``data_channels`` (may be any subset of the store's channels);
    the full stack is read but only the requested channels are retained.
    """
    missing = [c for c in list(data_channels) + list(coord_channels)
               if c not in source_channels]
    if missing:
        raise ValueError(f"channels not in dataset: {missing}. available: {source_channels}")

    d_idx = [source_channels.index(c) for c in data_channels]
    c_idx = [source_channels.index(c) for c in coord_channels]
    X = images_np[:, d_idx]                        # (N, C_feat, H, W) raw — only requested channels
    coords = images_np[:, c_idx]                   # (N, C_coord, H, W) raw
    mean = X.mean(axis=(0, 2, 3)).astype("float32")
    std = X.std(axis=(0, 2, 3)).astype("float32")
    return X, coords, mean, std


class _CutoutTorch(Dataset):
    """Thin torch Dataset over normalized feature images, for get_dataloader."""
    def __init__(self, X, ids):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.ids = ids                       # parallel to X; batched alongside it

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.ids[i]


class CutoutDataset:
    """In-memory cutout dataset: raw feature images X, separate raw coord fields,
    ids, metadata, channel names.  Build with ``CutoutDataset.from_source(...)``.

    X is kept raw; normalization (per-feature z-score) is applied on demand in
    ``get_patches`` / ``get_dataloader``, so ``make_image_from_patches`` can show
    physical values.
    """

    def __init__(self, X, coords, ids, metadata, channel_names, coord_names, mean, std):
        self.X = X                        # (N, C_feat, H, W) raw features
        self.coords = coords              # (N, C_coord, H, W) raw XC/YC
        self.ids = ids                    # decoded image_id per row, parallel to X
        self.metadata = metadata          # full df, indexed by image_id
        self.channel_names = channel_names   # feature names (no XC/YC)
        self.coord_names = coord_names       # e.g. ["XC", "YC"]
        self.mean, self.std = mean, std      # per-feature z-score stats (C_feat,)

    @classmethod
    def from_source(cls, source=None, data_channels=None, coord_channels=("XC", "YC"),
                    subset=True, subsample_per_chunk=300, num_sample_chunks=30, n_workers=8):
        source = source or CutoutDataSource()
        if data_channels is None:
            raise ValueError("pass data_channels; see source.print_available_channels()")

        images, ids = _download(source, subset, subsample_per_chunk, num_sample_chunks, n_workers)
        images, ids = _filter_invalid(images, ids, source.channel_names)
        X, coords, mean, std = _split_channels(
            images, source.channel_names, data_channels, coord_channels)
        print(f"features {list(data_channels)} | coords {list(coord_channels)}")
        return cls(X, coords, [_as_str(i) for i in ids], source.read_metadata(),
                   list(data_channels), list(coord_channels), mean, std)

    def __len__(self):
        return len(self.X)

    def _normalized(self, X=None):
        """Per-feature z-score.  Operates on the given array (default self.X) and
        computes the stats from it, so it composes after other transforms
        (e.g. after log-scaling the gradient fields)."""
        X = self.X if X is None else X
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        std = X.std(axis=(0, 2, 3), keepdims=True)
        return (X - mean) / std

    def _log_gradients(self, X=None):
        """log10 the grad-magnitude channels (name starts with 'grad'); other
        channels pass through unchanged.  Returns a copy."""
        X = (self.X if X is None else X).astype("float32", copy=True)
        for i, c in enumerate(self.channel_names):
            if c.startswith(_GRAD_PREFIX):
                X[:, i] = _safe_log10(X[:, i])
        return X

    def preprocess_for_training(self, X=None):
        """Full training transform: log-scale the gradient fields, then z-score.
        Order matters -- the z-score stats are computed on the logged data."""
        X = self.X if X is None else X
        return self._normalized(self._log_gradients(X))

    def get_patches(self, patch_size, flatten=True, preproc=True):
        """Feature patches for clustering.
        preproc=True (default) applies preprocess_for_training (log grads + z-score);
        preproc=False uses the raw stored X.
        flatten=True  -> (N_patches, C_feat*p*p) ndarray (clustering-ready)
        flatten=False -> (N_patches, C_feat, p, p) ndarray
        """
        X = self.preprocess_for_training() if preproc else self.X
        p = _to_patches(X, patch_size)
        return p.reshape(p.shape[0], -1) if flatten else p

    def get_patch_features(self, patch_size, preproc=False, log_grads=False):
        """Per-patch mean of each feature channel, (N_patches, C_feat), aligned
        with get_patches order.  preproc=False keeps physical units; preproc=True
        applies the training transform.  Reduces straight from X, so the full
        patch stack is never materialized.

        log_grads log10s the gradient-magnitude channels first, so those come back
        as a mean of logs, matching the training transform; it is redundant under
        preproc, which already logs them."""
        if preproc:
            X = self.preprocess_for_training()
        elif log_grads:
            X = self._log_gradients()
        else:
            X = self.X
        return reduce(X, 'n c (h p1) (w p2) -> (n h w) c', 'mean',
                      p1=patch_size, p2=patch_size)

    @property
    def log_scaled_channels(self):
        """Feature channels that the log-gradient transform log10s."""
        return [c for c in self.channel_names if c.startswith(_GRAD_PREFIX)]

    def get_patch_coords(self, patch_size):
        """Per-patch center (lon, lat), aligned with get_patches order."""
        c = _to_patches(self.coords, patch_size)
        xc, yc = self.coord_names.index("XC"), self.coord_names.index("YC")
        return c[:, xc].mean(axis=(-1, -2)), c[:, yc].mean(axis=(-1, -2))

    def get_patch_times(self, patch_size):
        """Per-patch timestamp (its cutout's), aligned with get_patches order."""
        H, W = self.X.shape[2], self.X.shape[3]
        ppi = (H // patch_size) * (W // patch_size)
        return np.repeat(self.metadata["time_snapshot"].reindex(self.ids).values, ppi)

    def get_dataloader(self, batch_size=64, shuffle=False, num_workers=0, preproc=True):
        """torch DataLoader yielding (feature images, ids) per batch.  preproc=True
        applies preprocess_for_training (same transform as get_patches); ids let you
        associate metadata (dataset.metadata.loc[ids]) during training."""
        X = self.preprocess_for_training() if preproc else self.X
        return DataLoader(_CutoutTorch(X, self.ids), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers, pin_memory=True)
