"""Coloring an embedding: the row mapping that makes it possible, and the
guard for when it is skipped."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from fronts_dataloader.fronts_dataset import FrontDataSource, RawFronts
from nemi_sweep.sweep import run_sweep
from visualization.embedding_plots import (
    CATEGORICAL_MAX, _is_categorical, _limits, plot_embedding_by,
    plot_embedding_features)
from visualization.sweep_plots import plot_embedding_grid

from synthetic_fronts import build_store


@pytest.fixture
def raw(tmp_path):
    build_store(tmp_path / "fronts.zarr", n=60)
    return RawFronts.load(FrontDataSource(str(tmp_path / "fronts.zarr")))


@pytest.fixture
def ds(raw):
    return raw.select(["geometry"], nan_policy="fill", missing_indicator=False)


# ---------------------------------------------------------------------------
#  Categorical or gradient

def test_cluster_labels_read_as_classes():
    assert _is_categorical(np.array([-1, 0, 1, 2, 2, 0]))


def test_many_whole_numbers_read_as_a_gradient():
    assert not _is_categorical(np.arange(CATEGORICAL_MAX + 1, dtype=float))


def test_continuous_values_read_as_a_gradient():
    assert not _is_categorical(np.linspace(-70, 70, 50))


def test_nan_does_not_decide_the_type():
    values = np.array([0.0, 1.0, np.nan, 2.0])
    assert _is_categorical(values)


# ---------------------------------------------------------------------------
#  Color limits

def test_limits_clip_the_tail():
    """The color scale follows the bulk, not the largest value present."""
    values = np.concatenate([np.random.default_rng(0).normal(size=999), [1e9]])
    _, hi = _limits(values)
    assert hi < 10


def test_limits_fall_back_when_the_clip_collapses():
    """Mass on one value plus a tail: clipping gives nothing, so use the range."""
    values = np.concatenate([np.zeros(999), [1e9]])
    assert _limits(values) == (0.0, 1e9)


def test_constant_column_has_no_limits():
    assert _limits(np.ones(100)) == (None, None)


def test_all_nan_has_no_limits():
    assert _limits(np.full(10, np.nan)) == (None, None)


# ---------------------------------------------------------------------------
#  The row mapping

def test_row_index_spans_the_table_when_nothing_drops(ds, raw):
    assert len(ds.row_index) == len(ds.X) == len(raw.table)


def test_row_index_tracks_dropped_rows(raw):
    dropped = raw.select(["geometry"], nan_policy="drop_rows")
    assert len(dropped.row_index) == len(dropped.X) < len(raw.table)


def test_meta_lines_up_with_the_matrix(raw):
    dropped = raw.select(["geometry"], nan_policy="drop_rows")
    meta = dropped.meta(raw.table, ["length_km"])
    column = dropped.feature_names.index("length_km")
    assert np.allclose(meta["length_km"], dropped.raw[:, column])


def test_meta_rejects_a_column_the_table_lacks(ds, raw):
    with pytest.raises(KeyError, match="not_a_column"):
        ds.meta(raw.table, ["not_a_column"])


def test_sweep_row_index_matches_its_embeddings(ds):
    result = run_sweep(ds.X, {"n_neighbors": [5]}, n_components=2,
                       device="cpu", metric_sample=20, keep_size=25,
                       verbose=False)
    assert len(result.row_index) == len(result.embeddings[0]) == 25
    assert result.row_index.max() < len(ds.X)


# ---------------------------------------------------------------------------
#  Drawing

def test_plot_by_rejects_a_mismatched_length():
    embedding = np.random.default_rng(0).normal(size=(50, 2))
    with pytest.raises(ValueError, match="row_index"):
        plot_embedding_by(embedding, np.arange(200))


def test_plot_features_rejects_an_unknown_column():
    embedding = np.random.default_rng(0).normal(size=(30, 2))
    frame = pd.DataFrame({"lat": np.linspace(-60, 60, 30)})
    with pytest.raises(KeyError, match="depth"):
        plot_embedding_features(embedding, frame, ["depth"])


def test_plot_by_draws_both_kinds():
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(60, 3))
    assert plot_embedding_by(embedding, rng.normal(size=60), dims=3)
    assert plot_embedding_by(embedding, rng.integers(-1, 3, 60), name="cluster")


def test_plot_features_makes_one_panel_per_column():
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(60, 2))
    frame = pd.DataFrame({"lat": rng.uniform(-70, 70, 60),
                          "branches": rng.integers(0, 4, 60),
                          "length": rng.uniform(5, 500, 60)})
    fig = plot_embedding_features(embedding, frame)
    assert len(fig.axes) >= len(frame.columns)


def test_color_by_beats_cluster_labels(ds, raw):
    """A clustering sweep still honors an explicit color_by."""
    result = run_sweep(ds.X, {"n_neighbors": [5], "min_dist": [0.0, 0.1]},
                       {"method": ["hdbscan"], "min_cluster_size": [5],
                        "min_samples": [None]},
                       n_components=2, device="cpu", metric_sample=20,
                       keep_size=25, verbose=False)
    assert any(lab is not None for lab in result.labels)
    lat = ds.meta(raw.table, ["centroid_lat"]).to_numpy().ravel()
    fig = plot_embedding_grid(result, x="n_neighbors", y="min_dist", dims=2,
                              color_by=lat[result.row_index])
    assert fig.axes
