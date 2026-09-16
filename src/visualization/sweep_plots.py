"""Plot a NEMI parameter sweep: metrics as heatmaps, embeddings as a grid.

Both put the same two swept parameters on the same axes, so a heatmap cell and
the scatter below it are the same configuration.  Smallest values sit
bottom-left in each.

    result = run_sweep(X, grid, cluster_grid)
    plot_all_metric_heatmaps(result, x="n_neighbors", y="min_dist")
    plot_embedding_grid(result, x="n_neighbors", y="min_dist")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#: Metrics whose natural reading is "fewer is better", so their colour scales
#: are reversed and the best cell is still the brightest.
_LOWER_IS_BETTER = ("normalized_stress", "noise_%", "top_cluster_%")

#: Integers read badly with three decimals.
_FMT = {"found_k": "{:.0f}", "noise_%": "{:.1f}", "top_cluster_%": "{:.1f}",
        "effective_k": "{:.1f}"}


def _frame(result):
    """The table, whether given a SweepResult or a DataFrame."""
    return result.table if hasattr(result, "table") else pd.DataFrame(result)


def _axis_values(column):
    """Unique values, sorted when orderable and left in sweep order when not.

    A parameter can hold None or a string (method names, min_samples=None),
    which sorted() refuses to compare.
    """
    values = list(dict.fromkeys(column))
    try:
        return sorted(values)
    except TypeError:
        return values


def _check_axes(df, *names):
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"{missing} not in results; available: {list(df.columns)}")


def plot_metric_heatmap(result, x, y, metric, *, ax=None, cmap="viridis",
                        annot=True, panel_size=4):
    """One metric over two swept parameters.

    Rows sharing an (x, y) cell are averaged, so where a third parameter also
    varies a cell is the mean over its values rather than a single run.
    """
    df = _frame(result)
    _check_axes(df, x, y, metric)

    grid = df.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
    values = grid.to_numpy(dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(panel_size * 1.35, panel_size))
    im = ax.imshow(values, cmap=f"{cmap}_r" if metric in _LOWER_IS_BETTER
                   else cmap, origin="lower", aspect="auto")
    ax.set_xticks(range(len(grid.columns)),
                  [str(c) for c in grid.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(metric + (" (lower is better)" if metric in _LOWER_IS_BETTER
                           else ""))
    ax.figure.colorbar(im, ax=ax, shrink=0.85)

    if annot:
        fmt = _FMT.get(metric, "{:.3f}")
        mid = np.nanmean(values) if np.isfinite(values).any() else 0.0
        for j in range(values.shape[0]):
            for i in range(values.shape[1]):
                v = values[j, i]
                if np.isfinite(v):
                    ax.text(i, j, fmt.format(v), ha="center", va="center",
                            fontsize=8,
                            color="white" if v < mid else "black")
    return ax


def plot_all_metric_heatmaps(result, x, y, *, metrics=None, cmap="viridis",
                             panel_size=4, ncols=3):
    """A heatmap per metric the sweep recorded, on shared axes."""
    df = _frame(result)
    if metrics is None:
        metrics = (result.metrics() if hasattr(result, "metrics")
                   else [c for c in df.columns if c not in (x, y)])
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise KeyError("no metric columns to plot")

    ncols = min(ncols, len(metrics))
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel_size * 1.35 * ncols,
                                      panel_size * nrows))
    for ax, metric in zip(axes.ravel(), metrics):
        plot_metric_heatmap(result, x, y, metric, ax=ax, cmap=cmap,
                            panel_size=panel_size)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_embedding_grid(result, x, y, *, dims=3, panel_size=3.5,
                        point_size=2, alpha=0.5, colour_by=None,
                        annot=True, annot_metrics=3):
    """The swept embeddings on the same axes as the heatmaps.

    One scatter per (x, y) cell instead of one colour.  Where a third parameter
    also varies, several configurations share a cell -- the heatmaps average
    them; here the first is drawn.

    Points are coloured by cluster label where the sweep clustered, else by
    *colour_by* (an array parallel to the embedding), else plain.
    """
    df = _frame(result)
    _check_axes(df, x, y)
    embeddings = getattr(result, "embeddings", None)
    if not embeddings:
        raise ValueError("this result carries no embeddings to plot")
    labels = getattr(result, "labels", [None] * len(df))

    xs, ys = _axis_values(df[x]), _axis_values(df[y])
    cells = {}
    for i, (_, row) in enumerate(df.iterrows()):
        cells.setdefault((row[y], row[x]), i)

    metrics = [m for m in (result.metrics() if hasattr(result, "metrics")
                           else []) if m in df.columns][:annot_metrics]

    fig, axes = plt.subplots(
        len(ys), len(xs), squeeze=False,
        figsize=(panel_size * len(xs), panel_size * len(ys)),
        subplot_kw={"projection": "3d" if dims == 3 else None})

    for row_i, yv in enumerate(ys):
        for col_i, xv in enumerate(xs):
            ax = axes[row_i][col_i]
            index = cells.get((yv, xv))
            if index is None:
                ax.axis("off")
                continue
            E = np.asarray(embeddings[index])
            colour = labels[index] if labels[index] is not None else colour_by
            ax.scatter(*[E[:, d] for d in range(min(dims, E.shape[1]))],
                       c=colour, s=point_size, alpha=alpha, cmap="tab20",
                       linewidths=0)
            title = f"{x}={xv}  {y}={yv}"
            if annot and metrics:
                title += "\n" + "  ".join(
                    f"{m}={df.iloc[index][m]:.2f}" for m in metrics
                    if pd.notna(df.iloc[index][m]))
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
            if dims == 3:
                ax.set_zticks([])
    fig.tight_layout()
    return fig
