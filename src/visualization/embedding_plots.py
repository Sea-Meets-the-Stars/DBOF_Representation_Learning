"""Color one embedding by whatever you want to see in it.

The sweep plots ask which configuration to keep; these ask what the kept one
is organized by.  One embedding, one panel per variable, so a gradient across
the cloud shows up as a gradient instead of being inferred from cluster means.

    lat = ds.meta(raw.table, "centroid_lat").to_numpy().ravel()
    plot_embedding_by(result.embeddings[0], lat[result.row_index],
                      name="centroid_lat")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from visualization.colors import NOISE, NOISE_COLOR, distinct_cmap

#: At or below this many distinct whole numbers a variable is drawn as classes
#: with a legend rather than a gradient with a colorbar.
CATEGORICAL_MAX = 12

#: Percentiles the color scale clips to.  Front statistics are heavy-tailed
#: enough that a raw min/max leaves every point at one end of the scale.
CLIP = (2, 98)

def _is_categorical(values) -> bool:
    """Few distinct whole numbers -> classes; anything else -> a gradient."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size or not np.all(finite == np.round(finite)):
        return False
    return len(np.unique(finite)) <= CATEGORICAL_MAX


def _limits(values):
    """Color limits clipped to CLIP.

    A column concentrated enough that the clipped range collapses -- most of
    the mass on one value, the rest in a long tail -- falls back to the full
    range, which at least renders.  A constant column returns (None, None)
    and lets matplotlib pick.
    """
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return None, None
    lo, hi = np.percentile(finite, CLIP)
    if lo == hi:
        lo, hi = finite.min(), finite.max()
    return (None, None) if lo == hi else (float(lo), float(hi))


def _check_length(embedding, values):
    if len(values) != len(embedding):
        raise ValueError(
            f"{len(values)} values for {len(embedding)} embedding rows.  A "
            "sweep stores a subsample of the rows it fitted, so line the "
            "values up with result.row_index before plotting.")


def _classes(ax, coords, values, point_size, alpha):
    """One scatter per class, noise first so clusters draw over it."""
    classes = np.unique(values)
    palette = distinct_cmap(len(classes))
    scatter = None
    for i, value in enumerate(classes):
        mask = values == value
        scatter = ax.scatter(
            *[c[mask] for c in coords], s=point_size, alpha=alpha,
            color=NOISE_COLOR if value == NOISE else palette(i),
            label="noise" if value == NOISE else f"{int(value)}",
            linewidths=0, zorder=1 if value == NOISE else 2)
    ax.legend(markerscale=6, fontsize=7, framealpha=0.8, loc="best")
    return scatter


def _draw(ax, embedding, values, dims, point_size, alpha, cmap, rng,
          categorical=None):
    """Scatter in shuffled order, so no one snapshot paints over another."""
    order = rng.permutation(len(embedding))
    coords = [embedding[order, d] for d in range(dims)]
    values = values[order]
    if categorical is None:
        categorical = _is_categorical(values)
    if categorical:
        return _classes(ax, coords, values, point_size, alpha), True
    lo, hi = _limits(values)
    scatter = ax.scatter(*coords, c=values, s=point_size, alpha=alpha,
                         cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
    return scatter, False


def _axes(fig, subplot, dims):
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    ax = fig.add_subplot(*subplot, projection="3d" if dims == 3 else None)
    ax.set_xticks([]), ax.set_yticks([])
    if dims == 3:
        ax.set_zticks([])
    return ax


def plot_embedding_by(embedding, values, *, name="value", dims=2, ax=None,
                      point_size=2, alpha=0.5, cmap="viridis", seed=0,
                      categorical=None):
    """One embedding, colored by one variable.

    *values* is parallel to *embedding*'s rows and in the variable's own units;
    the color scale clips to CLIP rather than to its range.  Whole-number
    variables with few distinct values get a class legend instead.
    """
    embedding = np.asarray(embedding)
    values = np.asarray(values).ravel()
    _check_length(embedding, values)

    fig = ax.figure if ax is not None else plt.figure(figsize=(7, 5.5))
    if ax is None:
        ax = _axes(fig, (1, 1, 1), dims)
    scatter, categorical = _draw(ax, embedding, values, dims, point_size,
                                 alpha, cmap, np.random.default_rng(seed),
                                 categorical)
    ax.set_title(name, fontsize=10)
    if not categorical:
        fig.colorbar(scatter, ax=ax, shrink=0.8, label=name)
    return fig


def plot_embedding_features(embedding, frame, columns=None, *, dims=2,
                            n_cols=3, panel_size=3.6, point_size=2,
                            alpha=0.5, cmap="viridis", seed=0):
    """The same embedding once per column of *frame*.

    *frame* is a DataFrame whose rows are the embedding's rows -- physical
    units, not the scaled matrix, so each colorbar reads in the variable's
    own units.  Pass *columns* to plot a subset.
    """
    embedding = np.asarray(embedding)
    frame = pd.DataFrame(frame)
    _check_length(embedding, frame)
    columns = list(frame.columns if columns is None else columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{missing} not in the frame; have {list(frame.columns)}")

    n_cols = max(1, min(n_cols, len(columns)))
    n_rows = -(-len(columns) // n_cols)
    fig = plt.figure(figsize=(panel_size * 1.25 * n_cols, panel_size * n_rows))
    rng = np.random.default_rng(seed)
    for i, column in enumerate(columns):
        ax = _axes(fig, (n_rows, n_cols, i + 1), dims)
        values = frame[column].to_numpy(dtype="float64")
        scatter, categorical = _draw(ax, embedding, values, dims, point_size,
                                     alpha, cmap, rng)
        ax.set_title(column, fontsize=9)
        if not categorical:
            fig.colorbar(scatter, ax=ax, shrink=0.75)
    fig.tight_layout()
    return fig
