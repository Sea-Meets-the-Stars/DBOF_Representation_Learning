"""The sweep's contract: one row per configuration, embeddings fitted once."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from nemi_sweep.sweep import SweepResult, expand_grid, run_sweep
from visualization.sweep_plots import (plot_all_metric_heatmaps,
                                       plot_embedding_grid,
                                       plot_metric_heatmap)

EMBED = {"n_neighbors": [10, 30], "min_dist": [0.0]}
CLUSTER = {"method": ["hdbscan"], "min_cluster_size": [15, 40],
           "min_samples": [5]}


@pytest.fixture(scope="module")
def blobs():
    """Three well-separated blobs, so clustering has an unambiguous answer."""
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(c, 1.0, (120, 5))
                      for c in (0, 8, 16)]).astype("float32")


@pytest.fixture(scope="module")
def result(blobs):
    return run_sweep(blobs, EMBED, CLUSTER, n_components=2, device="cpu",
                     metric_sample=150, keep_size=None, verbose=False)


# ---------------------------------------------------------------------------
#  Expanding the grids
# ---------------------------------------------------------------------------

def test_expand_grid_is_full_factorial():
    assert expand_grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"},
        {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]


def test_an_absent_grid_still_runs_once():
    """No clustering grid must not mean no embeddings."""
    assert expand_grid(None) == [{}]
    assert expand_grid({}) == [{}]


# ---------------------------------------------------------------------------
#  Running a sweep
# ---------------------------------------------------------------------------

def test_one_row_per_embedding_times_clustering(result):
    assert len(result) == 2 * 2                  # n_neighbors x min_cluster_size


def test_each_embedding_is_fitted_once_and_reused(result):
    """Reusing the embedding across clustering configs is the point; fitting
    per row would multiply the run by the size of the clustering grid."""
    assert result.n_fitted == 2
    assert len(result.embeddings) == len(result)


def test_the_table_carries_the_parameters_beside_the_metrics(result):
    for column in ("n_neighbors", "min_cluster_size", "trustworthiness",
                   "found_k", "noise_%"):
        assert column in result.table.columns


def test_it_finds_the_clusters_that_are_there(result):
    assert set(result.table["found_k"]) == {3}


def test_swept_names_only_what_varied(result):
    """min_dist and method were fixed, so they are not worth an axis."""
    assert set(result.swept) == {"n_neighbors", "min_cluster_size"}


def test_embedding_only_sweep_records_no_cluster_metrics(blobs):
    res = run_sweep(blobs, EMBED, n_components=2, device="cpu",
                    metric_sample=150, keep_size=None, verbose=False)
    assert len(res) == 2
    assert "trustworthiness" in res.table.columns
    assert "found_k" not in res.table.columns
    assert all(lab is None for lab in res.labels)


def test_keep_size_bounds_what_is_held_for_plotting(blobs):
    res = run_sweep(blobs, {"n_neighbors": [10]}, n_components=2,
                    device="cpu", metric_sample=100, keep_size=50,
                    verbose=False)
    assert len(res.embeddings[0]) == 50


def test_a_failing_config_is_recorded_not_raised(blobs):
    """One bad configuration must not lose the rest of the sweep."""
    res = run_sweep(blobs, {"n_neighbors": [10]},
                    {"method": ["nonsense"]}, n_components=2, device="cpu",
                    metric_sample=100, keep_size=None, verbose=False)
    assert len(res) == 1                          # the row is still there
    assert res.failures
    assert "found_k" not in res.table.columns     # but it has no cluster scores


def test_best_orients_each_metric_correctly(result):
    hi = result.best("trustworthiness", n=1)["trustworthiness"].iloc[0]
    assert hi == result.table["trustworthiness"].max()
    lo = result.best("normalized_stress", n=1)["normalized_stress"].iloc[0]
    assert lo == result.table["normalized_stress"].min()


def test_best_rejects_a_metric_that_was_not_recorded(result):
    with pytest.raises(KeyError, match="silhouette"):
        result.best("silhouette")


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def test_a_heatmap_covers_every_swept_cell(result):
    ax = plot_metric_heatmap(result, "n_neighbors", "min_cluster_size",
                             "trustworthiness")
    assert ax.images
    assert ax.images[0].get_array().shape == (2, 2)
    plt.close(ax.figure)


def test_all_metrics_get_a_panel(result):
    fig = plot_all_metric_heatmaps(result, "n_neighbors", "min_cluster_size")
    drawn = [a for a in fig.axes if a.get_title()]
    assert {a.get_title().split(" (")[0] for a in drawn} >= set(result.metrics())
    plt.close(fig)


def test_the_embedding_grid_draws_one_panel_per_cell(result):
    fig = plot_embedding_grid(result, "n_neighbors", "min_cluster_size", dims=2)
    assert len([a for a in fig.axes if a.collections]) == 4
    plt.close(fig)


def test_plots_name_the_column_that_is_missing(result):
    with pytest.raises(KeyError, match="nope"):
        plot_metric_heatmap(result, "nope", "min_dist", "trustworthiness")


def test_the_embedding_grid_needs_embeddings():
    empty = SweepResult(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(ValueError, match="no embeddings"):
        plot_embedding_grid(empty, "a", "b")
