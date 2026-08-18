"""Local stand-ins for a cutout dataset on S3.

The image store is a real zarr group with the production layout written by
``dbof``'s ``ZarrDataset`` -- (N, C, H, W) images and (N,) ``S32`` image_ids,
one chunk per cutout -- and the metadata is real parquet written by dbof's
``MetadataWriter``.  Only the S3 filesystem is replaced, so ids come back as
``numpy.bytes_`` and unwritten slots read as ``b""`` exactly as they do in
production.

Three kinds of row are modelled, matching what a real store contains:

  written   image + id + metadata row   -- a good cutout
  orphan    image + id, no metadata row -- crashed before the metadata flushed
  empty     neither                     -- a cutout rejected at generation time
"""
import types

import dask.array as da
import fsspec
import numpy as np
import pandas as pd
import zarr

import dbof.cutout_dataset_creation.metadata as metadata_mod
import llc_cutout_dataloader.cutouts_dataset as cutouts_dataset

CHANNELS = ["Theta", "Salt", "SIarea", "XC", "YC"]
DATA_CHANNELS = ["Theta", "Salt"]
H = W = 8
FOLDER = "cutouts"
RUN_ID = "run0"
DATASET_NAME = "cutout_dataset.zarr"


def image_for(row: int, ice: bool = False, nan: bool = False) -> np.ndarray:
    """One cutout whose contents identify its row, so alignment is checkable.

    Theta carries the row number, XC/YC carry derived coordinates, and SIarea
    is ice-free unless asked otherwise.
    """
    img = np.zeros((len(CHANNELS), H, W), dtype="float32")
    img[CHANNELS.index("Theta")] = float(row)
    img[CHANNELS.index("XC")] = float(row) + 0.5
    img[CHANNELS.index("YC")] = -float(row) - 0.5
    if ice:
        img[CHANNELS.index("SIarea")] = 1.0
    if nan:
        img[CHANNELS.index("Salt")] = np.nan
    return img


def row_of(image: np.ndarray) -> int:
    """Recover the row number an image was built with (inverse of image_for)."""
    return int(image[DATA_CHANNELS.index("Theta")].flat[0])


def row_of_id(image_id) -> int:
    return int(cutouts_dataset._as_str(image_id).removeprefix("id"))


class FakeSource(cutouts_dataset.CutoutDataSource):
    """CutoutDataSource over local storage.

    ``__init__`` and ``full_dataset_as_dask`` are replaced; ``read_metadata``
    and ``source_info`` run the production implementations.  Keep
    ``full_dataset_as_dask`` in step with
    ``ZarrDatasetReader.full_dataset_as_dask``.
    """

    def __init__(self, tmp_path, n_written=6, n_orphan=2, n_empty=2,
                 ice_rows=(), nan_rows=()):
        # Metadata lives on a memory filesystem keyed by `bucket`, which
        # create_metadata_reader strips leading slashes from -- so it cannot be
        # an absolute path.  tmp_path.name keeps each test's tree separate.
        self.bucket = tmp_path.name
        self.folder, self.run_id = FOLDER, RUN_ID
        self.dataset_name, self.s3_endpoint = DATASET_NAME, "local"
        self.channel_names = list(CHANNELS)
        self.fs_synch = fsspec.filesystem("memory")
        self.reader = types.SimpleNamespace(
            down_sample_res=H, target_km_res=150, channel_names=list(CHANNELS))

        n = n_written + n_orphan + n_empty
        self.written_rows = list(range(n_written))
        self.orphan_rows = list(range(n_written, n_written + n_orphan))
        self.empty_rows = list(range(n_written + n_orphan, n))

        store = zarr.storage.LocalStore(str(tmp_path / self.dataset_name))
        self._root = zarr.open_group(store=store, mode="a")
        self._root.attrs["channel_names"] = list(CHANNELS)
        self._root.attrs["target_km_res"] = 150
        self._root.attrs["down_sample_res"] = H
        images = self._root.create_array(
            "images", shape=(n, len(CHANNELS), H, W),
            chunks=(1, len(CHANNELS), H, W), dtype="float32")
        ids = self._root.create_array("image_ids", shape=(n,), chunks=(1,), dtype="S32")

        # Empty rows are left untouched: no chunk is written, so the id reads
        # back as b"" -- the same state a rejected cutout leaves behind.
        for row in self.written_rows + self.orphan_rows:
            images[row] = image_for(row, ice=row in ice_rows, nan=row in nan_rows)
            ids[row] = f"id{row:02d}".encode("ascii")

        writer = metadata_mod.MetadataWriter(
            f"{self.bucket}/{self.folder}/{self.run_id}/metadata", fs=self.fs_synch)
        for row in self.written_rows:
            writer.add({
                "image_id": f"id{row:02d}".encode("ascii"),
                "center_lat": -float(row) - 0.5,
                "center_lon": float(row) + 0.5,
                "time_snapshot": pd.Timestamp("2012-01-01") + pd.Timedelta(days=row),
            })
        writer.close()

    def full_dataset_as_dask(self):
        images_da = da.from_zarr(self._root["images"])
        ids_da = da.from_zarr(self._root["image_ids"])
        return images_da.rechunk((1024, -1, -1, -1)), ids_da, ids_da != b""

    @property
    def expected_ids(self):
        """Ids _download should keep: written and carrying a metadata row."""
        return [f"id{row:02d}".encode("ascii") for row in self.written_rows]
