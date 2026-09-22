"""Colors for discrete labels, shared by every view that draws clusters.

Kept apart from visualization.py so the sweep and embedding plots can use it
without pulling in cartopy, cmocean and dbof.
"""
import colorsys

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

#: The unclustered label, and the grey it is drawn in.
NOISE = -1
NOISE_COLOR = (0.6, 0.6, 0.6, 1.0)


def distinct_cmap(n):
    """Qualitative colormap with n visually distinct colors, for any n."""
    return ListedColormap([
        colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0,
                            0.55 + 0.35 * (i % 2),
                            0.75 + 0.20 * ((i // 2) % 2)) for i in range(n)])


def cluster_cmap_norm(labels, cmap=None):
    """Discrete cmap and norm for integer labels, with a grey slot for NOISE.

    Passing a qualitative colormap to ``c=`` without a norm samples it
    linearly instead: -1 takes a cluster's color rather than grey, a panel
    with one cluster degenerates, and above about twenty labels adjacent
    clusters get the same RGB.
    """
    values = np.asarray(labels, dtype=float)
    finite = values[np.isfinite(values)]
    hi = int(finite.max()) if finite.size else 0
    lo = int(finite.min()) if finite.size else 0
    base = cmap or distinct_cmap(hi + 1)
    colors = [base(k) for k in range(hi + 1)]
    if lo < 0:
        colors = [NOISE_COLOR] + colors
    # NaN labels -- a NEMI member that clustered nothing -- read as noise too.
    out = ListedColormap(colors).with_extremes(bad=NOISE_COLOR)
    return out, BoundaryNorm(np.arange(min(lo, 0), hi + 2) - 0.5, len(colors))
