"""Tests for the cutout loading path, against a local store (see synthetic_cutouts)."""
import numpy as np
import pyarrow.parquet as pq
import pytest

import llc_cutout_dataloader.cutouts_dataset as cutouts_dataset
import synthetic_cutouts as synth
from synthetic_cutouts import DATA_CHANNELS, FakeSource, row_of, row_of_id


def _download(source, subset=False, subsample_per_chunk=5, num_sample_chunks=1,
              metadata=None):
    return cutouts_dataset._download(
        source, subset, subsample_per_chunk, num_sample_chunks, n_workers=1,
        metadata=metadata)


# ---------------------------------------------------------------------------
# _download: which rows survive
# ---------------------------------------------------------------------------

def test_download_drops_empty_slots_and_orphans(tmp_path):
    source = FakeSource(tmp_path, n_written=6, n_orphan=2, n_empty=2)

    images, ids = _download(source, metadata=source.read_metadata())

    assert list(ids) == source.expected_ids
    assert images.shape[0] == 6


def test_download_drops_orphans_even_when_no_slot_is_empty(tmp_path):
    """The orphan filter must not lean on the empty-slot mask to do its work."""
    source = FakeSource(tmp_path, n_written=4, n_orphan=3, n_empty=0)

    _, ids = _download(source, metadata=source.read_metadata())

    assert list(ids) == source.expected_ids


def test_download_reads_metadata_when_not_supplied(tmp_path):
    source = FakeSource(tmp_path, n_written=3, n_orphan=2, n_empty=1)

    _, ids = _download(source, metadata=None)

    assert list(ids) == source.expected_ids


# ---------------------------------------------------------------------------
# _download: images stay aligned to ids
# ---------------------------------------------------------------------------

def test_download_keeps_ids_aligned_to_images(tmp_path):
    source = FakeSource(tmp_path, n_written=6, n_orphan=2, n_empty=2)

    images, ids = _download(source, metadata=source.read_metadata())

    theta = synth.CHANNELS.index("Theta")
    for image, image_id in zip(images, ids):
        assert int(image[theta].flat[0]) == row_of_id(image_id)


def test_download_subset_keeps_ids_aligned_to_images(tmp_path):
    """Regression: the subset branch once re-indexed the *unfiltered* id array."""
    source = FakeSource(tmp_path, n_written=6, n_orphan=2, n_empty=2)

    images, ids = _download(source, subset=True, subsample_per_chunk=4,
                            num_sample_chunks=1, metadata=source.read_metadata())

    assert images.shape[0] == len(ids) == 4
    theta = synth.CHANNELS.index("Theta")
    for image, image_id in zip(images, ids):
        assert int(image[theta].flat[0]) == row_of_id(image_id)
        assert image_id in source.expected_ids


# ---------------------------------------------------------------------------
# _filter_invalid
# ---------------------------------------------------------------------------

def test_filter_invalid_drops_ice_and_nan_and_keeps_alignment(tmp_path):
    source = FakeSource(tmp_path, n_written=6, n_orphan=0, n_empty=0,
                        ice_rows=(1,), nan_rows=(3,))
    images, ids = _download(source, metadata=source.read_metadata())

    kept_images, kept_ids = cutouts_dataset._filter_invalid(
        images, ids, source.channel_names)

    assert [row_of_id(i) for i in kept_ids] == [0, 2, 4, 5]
    theta = synth.CHANNELS.index("Theta")
    for image, image_id in zip(kept_images, kept_ids):
        assert int(image[theta].flat[0]) == row_of_id(image_id)


# ---------------------------------------------------------------------------
# from_source
# ---------------------------------------------------------------------------

def test_from_source_ids_are_all_present_in_metadata(tmp_path):
    """The invariant a downstream label join depends on: every id resolves."""
    source = FakeSource(tmp_path, n_written=6, n_orphan=3, n_empty=2)

    dataset = cutouts_dataset.CutoutDataset.from_source(
        source=source, data_channels=DATA_CHANNELS, subset=False)

    assert set(dataset.ids) <= set(dataset.metadata.index)
    assert len(dataset.ids) == 6


def test_from_source_keeps_requested_channels_and_separates_coords(tmp_path):
    source = FakeSource(tmp_path, n_written=4, n_orphan=0, n_empty=0)

    dataset = cutouts_dataset.CutoutDataset.from_source(
        source=source, data_channels=DATA_CHANNELS, subset=False)

    assert dataset.channel_names == DATA_CHANNELS
    assert dataset.X.shape == (4, len(DATA_CHANNELS), synth.H, synth.W)
    assert dataset.coords.shape == (4, 2, synth.H, synth.W)
    for row, image in enumerate(dataset.X):
        assert row_of(image) == row
    xc = dataset.coords[:, dataset.coord_names.index("XC")]
    assert np.allclose(xc[:, 0, 0], np.arange(4) + 0.5)


def test_from_source_rejects_unknown_channel(tmp_path):
    source = FakeSource(tmp_path, n_written=2, n_orphan=0, n_empty=0)

    with pytest.raises(ValueError, match="channels not in dataset"):
        cutouts_dataset.CutoutDataset.from_source(
            source=source, data_channels=["Theta", "Nope"], subset=False)


def test_patch_times_resolve_for_every_patch(tmp_path):
    """get_patch_times reindexes metadata by id; an unjoinable id yields NaT."""
    source = FakeSource(tmp_path, n_written=4, n_orphan=2, n_empty=0)
    dataset = cutouts_dataset.CutoutDataset.from_source(
        source=source, data_channels=DATA_CHANNELS, subset=False)

    times = dataset.get_patch_times(patch_size=4)

    assert len(times) == 4 * 4          # 4 cutouts x 4 patches
    assert not np.isnat(times).any()


# ---------------------------------------------------------------------------
# label round-trip (what a downstream consumer joins against)
# ---------------------------------------------------------------------------

def test_save_cluster_labels_keys_every_row_to_metadata(tmp_path):
    source = FakeSource(tmp_path, n_written=4, n_orphan=2, n_empty=1)
    dataset = cutouts_dataset.CutoutDataset.from_source(
        source=source, data_channels=DATA_CHANNELS, subset=False)
    patch_size = 4
    ppi = (synth.H // patch_size) * (synth.W // patch_size)
    out = tmp_path / "labels.parquet"

    dataset.save_cluster_labels(str(out), np.arange(len(dataset.ids) * ppi), patch_size)

    table = pq.read_table(out)
    assert set(table.column("image_id").to_pylist()) <= set(dataset.metadata.index)
    assert table.num_rows == len(dataset.ids)
    assert len(table.column("labels")[0]) == ppi


def test_save_cluster_labels_rejects_mismatched_label_count(tmp_path):
    source = FakeSource(tmp_path, n_written=3, n_orphan=0, n_empty=0)
    dataset = cutouts_dataset.CutoutDataset.from_source(
        source=source, data_channels=DATA_CHANNELS, subset=False)

    with pytest.raises(ValueError, match="different runs"):
        dataset.save_cluster_labels(str(tmp_path / "bad.parquet"),
                                    np.arange(5), patch_size=4)


# ---------------------------------------------------------------------------
# subsampling helper
# ---------------------------------------------------------------------------

def test_chunk_aware_subsample_stays_in_range(tmp_path):
    source = FakeSource(tmp_path, n_written=6, n_orphan=0, n_empty=0)
    images_da, _, _ = source.full_dataset_as_dask()

    idx = cutouts_dataset.chunk_aware_subsample(
        images_da, num_sample_chunks=1, subsample_per_chunk=5)

    assert idx.size == 5
    assert idx.min() >= 0 and idx.max() < images_da.shape[0]
    assert (np.diff(idx) >= 0).all()
