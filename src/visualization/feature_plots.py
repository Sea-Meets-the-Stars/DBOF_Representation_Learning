"""What each feature looked like before and after the loader transformed it.

Column 0 is the value as the store holds it; column 1 is what reaches the
clusterer, titled with the chain that produced it.  Reuses the robust limits
and the grid layout from the embedding plots, so a heavy tail does not put
every front in one bin.
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from visualization.embedding_plots import _limits


def plot_feature_transforms(ds, table, columns=None, bins=60,
                            panel=(5.0, 3.0)):
    """Raw and transformed distribution per feature, one row per feature.

    Parameters
    ----------
    ds : FrontDataset
    table : pandas.DataFrame
        The RawFronts table *ds* was built from; the untransformed values come
        from here, on ``ds.row_index`` so both panels hold the same fronts.
    columns : sequence, optional
        Features to draw.  Defaults to every one in the dataset.
    bins : int
    panel : (float, float)
        Width and height of one panel, in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    columns = list(ds.feature_names if columns is None else columns)
    missing = [c for c in columns if c not in ds.feature_names]
    if missing:
        raise KeyError(f"{missing} not in the dataset; it has "
                       f"{len(ds.feature_names)} features")

    fig, axes = plt.subplots(len(columns), 2, squeeze=False,
                             figsize=(2 * panel[0], len(columns) * panel[1]))
    for row, col in enumerate(columns):
        j = ds.feature_names.index(col)
        before = table[col].to_numpy(float)[ds.row_index]
        after = np.asarray(ds.raw[:, j], dtype=float)
        for ax, v, title in ((axes[row][0], before, f"{col}  (as stored)"),
                             (axes[row][1], after,
                              f"{col}\n{ds.transform_label(col)}")):
            finite = v[np.isfinite(v)]
            lo, hi = _limits(finite)
            ax.hist(finite, bins=bins, range=None if lo is None else (lo, hi),
                    color="steelblue", edgecolor="white", linewidth=0.3)
            ax.set_title(title, fontsize=9)
            # Log counts: a heavy tail is one visible bar otherwise, and both
            # panels share the scale so the pair stays comparable.
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if lo is not None:
                outside = int(finite.size
                              - ((finite >= lo) & (finite <= hi)).sum())
                if outside:
                    ax.set_xlabel(f"{outside:,} outside the 2-98 pct range",
                                  fontsize=8, color="0.45")
    fig.tight_layout()
    return fig
