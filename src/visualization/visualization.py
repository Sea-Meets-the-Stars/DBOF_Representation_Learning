import colorsys
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from dbof.plotting.field_cmaps import load_field_cmaps

try:
    _FIELD_CMAPS, _DIVERGING_CMAPS = load_field_cmaps()   # {channel: (cmocean_name, label)}
except Exception:
    _FIELD_CMAPS, _DIVERGING_CMAPS = {}, set()

def _make_ax(fig, dims, subplot=(1, 1, 1)):
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    return fig.add_subplot(*subplot, projection="3d" if dims == 3 else None)


def _scatter_embedding(ax, X_d, labels=None, dims=2, alpha=0.5, s=0.1, cmap=None, norm=None):
    kw = dict(s=s, alpha=alpha)
    if labels is not None:
        kw.update(c=labels, cmap=cmap, norm=norm)
    scatter = ax.scatter(*[X_d[:, i] for i in range(dims)], **kw)
    ax.set_xlabel("C1")
    ax.set_ylabel("C2")
    if dims == 3:
        ax.set_zlabel("C3")
    return scatter

def _set_limits(ax, X_d, dims, pct=1.0):
    """Frame the central (100 - 2*pct)% per axis, ignoring outliers."""
    setters = [ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if dims == 3 else [])
    for i, setlim in enumerate(setters):
        lo, hi = np.percentile(X_d[:, i], [pct, 100 - pct])
        setlim(lo, hi)

def _add_class_legend(ax, scatter, label_title):
    ax.add_artist(ax.legend(*scatter.legend_elements(), title=label_title))


def _annotate(fig, ax, scatter, categorical, label_title):
    """Categorical -> discrete class legend; numerical -> colorbar."""
    if categorical:
        _add_class_legend(ax, scatter, label_title)
    else:
        fig.colorbar(scatter, ax=ax, label=label_title)


def _labels_per_embedding(labels, n):
    """None, a shared (N,) array, or n per-embedding label arrays -> list of length n."""
    if labels is None:
        return [None] * n
    if len(labels) == n:            # (n, N) array or list of n label vectors
        return list(labels)
    return [labels] * n             # single shared (N,) array


def vis_dim_redux(X_d, labels=None, categorical=True, label_title="class", dims=2, alpha=0.5, cmap=None):
    norm = None
    if labels is not None and categorical:
        cmap, norm = _cluster_cmap_norm(labels, cmap)
    else:
        cmap = cmap or "viridis"
    fig = plt.figure(figsize=(8, 6))
    ax = _make_ax(fig, dims)
    scatter = _scatter_embedding(ax, X_d, labels=labels, dims=dims, alpha=alpha, cmap=cmap, norm=norm)
    if labels is not None:
        _annotate(fig, ax, scatter, categorical, label_title)
    _set_limits(ax, X_d, dims, pct=1.0)
    plt.show()


def vis_dim_redux_list(embeddings, labels=None, categorical=True, titles=None, label_title="class",
                       dims=2, alpha=0.5, n_cols=3, cmap=None):
    """Scatter a list/array of embeddings in a grid.

    embeddings : iterable of (N, dims) arrays, or an (M, N, dims) array.
    labels     : None, a shared (N,) array applied to every panel, or one label
                 array per embedding (e.g. member_clusters of shape (M, N)).
    titles     : optional per-panel titles.
    """
    n = len(embeddings)
    n_cols = min(n_cols, n)
    n_rows = -(-n // n_cols)                      # ceil
    per_labels = _labels_per_embedding(labels, n)

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    for i, X_d in enumerate(embeddings):
        ax = _make_ax(fig, dims, (n_rows, n_cols, i + 1))
        lab = per_labels[i]
        if lab is not None and categorical:
            pc, pn = _cluster_cmap_norm(lab, cmap)
        else:
            pc, pn = (cmap or "viridis"), None
        scatter = _scatter_embedding(ax, X_d, labels=lab, dims=dims, alpha=alpha, cmap=pc, norm=pn)
        if titles is not None:
            ax.set_title(titles[i])
        if per_labels[i] is not None:
            _annotate(fig, ax, scatter, categorical, label_title)
        _set_limits(ax, X_d, dims, pct=1.0)
    fig.tight_layout()
    plt.show()


_DEFAULT_META_FIELDS = ("time_snapshot", "center_lat", "center_lon", "log_grad_b_2_center")


def distinct_cmap(n):
    """Qualitative colormap with n visually distinct colors (any n)."""
    colors = [colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0,
                                  0.55 + 0.35 * (i % 2),
                                  0.75 + 0.20 * ((i // 2) % 2)) for i in range(n)]
    return ListedColormap(colors)


_NOISE_COLOR = (0.6, 0.6, 0.6, 1.0)   # grey slot for -1 / NaN noise


def _cluster_cmap_norm(labels, cmap=None):
    """Discrete cmap + BoundaryNorm for integer cluster labels, with a grey slot
    for -1 (and NaN) noise.  Cluster k keeps its distinct_cmap(k) color, so colors
    match across the patch grid, embedding, and map views."""
    labels = np.asarray(labels, dtype=float)
    finite = labels[np.isfinite(labels)]
    hi = int(finite.max()) if finite.size else 0
    lo = int(finite.min()) if finite.size else 0
    base = cmap or distinct_cmap(hi + 1)
    colors = [base(k) for k in range(hi + 1)]
    bounds = np.arange(lo if lo < 0 else 0, hi + 2) - 0.5
    if lo < 0:
        colors = [_NOISE_COLOR] + colors      # -1 -> grey
    out = ListedColormap(colors)
    out.set_bad(_NOISE_COLOR)                  # NaN (member noise) -> grey too
    return out, BoundaryNorm(bounds, len(colors))


def _as_int_labels(labels):
    """Cluster labels as ints, with NaN noise (single-member NEMI) folded into -1,
    so -1 and NaN noise are one cluster everywhere downstream."""
    return np.nan_to_num(np.asarray(labels, dtype=float), nan=-1.0).astype(int)


def _cluster_order(labels, drop_noise=False):
    """(ids, counts, fractions) per cluster, ordered by descending size.  Noise
    (-1) is a cluster like any other unless drop_noise; fractions are of the
    retained labels, so they always sum to 1."""
    labels = _as_int_labels(labels)
    if drop_noise:
        labels = labels[labels >= 0]
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts, kind="stable")     # ties keep ascending id
    ids, counts = ids[order], counts[order]
    return ids, counts, counts / counts.sum()


def _select_clusters(labels, top_n=None, n_clusters=20, drop_noise=False, seed=0):
    """Clusters to plot, as (ids, counts, percent).

    top_n given -> the largest n, size-ordered.  Otherwise n_clusters drawn at
    random from every cluster present (so small clusters are represented too)
    and returned in ascending label order.
    """
    ids, counts, frac = _cluster_order(labels, drop_noise)
    if top_n is not None:
        keep = np.arange(min(top_n, ids.size))
    else:
        keep = np.random.default_rng(seed).choice(
            ids.size, size=min(n_clusters, ids.size), replace=False)
        keep = keep[np.argsort(ids[keep])]
    return ids[keep], counts[keep], 100 * frac[keep]


def _resolve_features(features, channel_names):
    """None -> all channels; else list of names/indices -> [(idx, name), ...]."""
    if features is None:
        return [(i, channel_names[i]) for i in range(len(channel_names))]
    out = []
    for f in features:
        idx = channel_names.index(f) if isinstance(f, str) else int(f)
        out.append((idx, channel_names[idx]))
    return out


def _field_style(name, data):
    """(cmap, vmin, vmax) for a field from the llc4320 registry: diverging fields
    (balance/curl) centered at zero, others on robust 1-99 percentiles.  Falls
    back to the matplotlib default cmap for channels not in the registry."""
    spec = _FIELD_CMAPS.get(name)
    cmap = getattr(cmocean.cm, spec[0], None) if spec else None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return cmap, None, None
    if spec and spec[0] in _DIVERGING_CMAPS:
        m = float(np.abs(finite).max())
        return cmap, -m, m
    lo, hi = np.percentile(finite, [1, 99])
    return cmap, float(lo), float(hi)


def _field_label(name):
    """Physical label from the llc4320 registry, falling back to the raw name."""
    spec = _FIELD_CMAPS.get(name)
    return spec[1] if spec else name


def _axis_label(name, logged=False, standardized=False):
    """Axis label for a feature channel, marked with the transforms applied."""
    label = _field_label(name)
    if logged:
        label = f"log10 {label}"
    return f"{label} (z-scored)" if standardized else label


def _format_meta(row, fields):
    if row is None:
        return ""
    fmt = {"center_lat": ("lat", "{:.2f}"), "center_lon": ("lon", "{:.2f}"),
           "log_grad_b_2_center": ("log|∇b|²(c)", "{:.2f}"),
           "time_snapshot": ("time", None)}
    parts = []
    for f in fields:
        if f not in row:
            continue
        label, spec = fmt.get(f, (f, None))
        val = (str(np.datetime64(row[f], "s")) if f == "time_snapshot"
               else spec.format(float(row[f])) if spec else str(row[f]))
        parts.append(f"{label}={val}")
    return "     ".join(parts)


_ENTROPY_CMAP = "viridis"


def _patch_grid(ax, H, W, patch_size):
    """Patch boundaries over the current image."""
    for g in range(0, H + 1, patch_size):
        ax.axhline(g, color="k", lw=0.8, alpha=0.6)
    for g in range(0, W + 1, patch_size):
        ax.axvline(g, color="k", lw=0.8, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])


def _patch_overlay(ax, bg, bg_style, values, patch_size, *,
                   cmap, norm=None, vmin=None, vmax=None):
    """Field background with one value per patch drawn over the patch grid.

    bg_style : (cmap, vmin, vmax) for the background, from _field_style.
    values   : flat per-patch array for this cutout, row-major.
    """
    H, W = bg.shape
    bg_cmap, bg_vmin, bg_vmax = bg_style
    ax.imshow(bg, extent=[0, W, H, 0], cmap=bg_cmap, vmin=bg_vmin, vmax=bg_vmax)
    im = ax.imshow(values.reshape(H // patch_size, W // patch_size),
                   cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=0.9,
                   extent=[0, W, H, 0], interpolation="nearest")
    _patch_grid(ax, H, W, patch_size)
    return im


def _highlight_overlay(ax, bg, bg_style, mask, patch_size, color, alpha=0.55):
    """Field background with only the patches in ``mask`` tinted, for inspecting
    one cluster.  Everything else is left as the bare field."""
    H, W = bg.shape
    bg_cmap, bg_vmin, bg_vmax = bg_style
    ax.imshow(bg, extent=[0, W, H, 0], cmap=bg_cmap, vmin=bg_vmin, vmax=bg_vmax)
    rgba = np.zeros((H // patch_size, W // patch_size, 4))
    rgba[mask.reshape(H // patch_size, W // patch_size)] = (*color[:3], alpha)
    ax.imshow(rgba, extent=[0, W, H, 0], interpolation="nearest")
    _patch_grid(ax, H, W, patch_size)


def make_image_from_patches(dataset, labels, *, entropy=None,
                            features=None, label_overlay_on=0,
                            patch_size=8, start=0, number_rows=6,
                            cutouts=None, highlight=None, show_map=False,
                            metadata_fields=_DEFAULT_META_FIELDS,
                            panel_size=3, cmap=None,
                            meta_fontsize=None, title_fontsize=None, save_path=None):
    """Grid of cutouts: one panel per selected feature (titled with its zarr
    channel name, colored with its llc4320 field colormap) + a cluster-label
    overlay, and a metadata caption under each row.

    dataset          : CutoutDataset (provides raw X, channel_names, ids, metadata).
    entropy          : optional per-patch ensemble entropy (NEMI.entropy, normalized
                       to [0, 1]); adds a panel beside the cluster labels, on a
                       fixed [0, 1] scale so rows and figures are comparable.
    features         : channel names/indices to show (default: all features).
    label_overlay_on : channel name/index used as the overlay background.
    cutouts          : cutout indices to show, in order (default: every cutout).
    start            : offset into that order for the first row shown.
    highlight        : cluster label; the overlay then tints only that cluster's
                       patches and leaves the rest of the field bare.
    show_map         : add a column locating each cutout on a global map.
    save_path        : if given, save the figure there before showing.

    ``labels`` (and ``entropy``) are flat per-patch in dataset order
    (patch k -> cutout k // ppi).
    """
    channel_names = dataset.channel_names
    imgs, metadata, ids = dataset.X, dataset.metadata, dataset.ids
    feats = _resolve_features(features, channel_names)
    bg_idx = _resolve_features([label_overlay_on], channel_names)[0][0]

    meta_fontsize = meta_fontsize or int(panel_size * 6)
    title_fontsize = title_fontsize or int(panel_size * 4)

    rows = np.arange(len(imgs)) if cutouts is None else np.asarray(cutouts, dtype=int)
    rows = rows[start:start + number_rows]
    number_rows = len(rows)
    H, W = imgs.shape[2], imgs.shape[3]
    n_h, n_w = H // patch_size, W // patch_size
    ppi = n_h * n_w

    labels = _as_int_labels(labels)
    cmap, norm = _cluster_cmap_norm(labels, cmap)   # grey slot for -1 noise

    if entropy is not None:
        entropy = np.asarray(entropy)
        assert entropy.size == labels.size, \
            f"{entropy.size} entropy values vs {labels.size} labels"

    n_col = len(feats) + 1 + (1 if entropy is not None else 0) + (1 if show_map else 0)
    fig = plt.figure(figsize=(panel_size * n_col, panel_size * 1.4 * number_rows))
    gs = fig.add_gridspec(number_rows * 2, n_col,
                          height_ratios=[6, 2] * number_rows, hspace=0.06, wspace=0.05)
    entropy_axes = []

    for r, c in enumerate(rows):
        img = np.asarray(imgs[c])
        sl = slice(c * ppi, (c + 1) * ppi)
        row = metadata.loc[ids[c]] if ids[c] in metadata.index else None

        for col, (idx, name) in enumerate(feats):
            ax = fig.add_subplot(gs[2 * r, col])
            f_cmap, vmin, vmax = _field_style(name, img[idx])
            ax.imshow(img[idx], cmap=f_cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=title_fontsize)

        bg_style = _field_style(channel_names[bg_idx], img[bg_idx])

        ax = fig.add_subplot(gs[2 * r, len(feats)])
        patch_labels = labels[sl]
        assert patch_labels.size == ppi, f"expected {ppi} labels, got {patch_labels.size}"
        if highlight is None:
            _patch_overlay(ax, img[bg_idx], bg_style, patch_labels, patch_size,
                           cmap=cmap, norm=norm)
        else:
            _highlight_overlay(ax, img[bg_idx], bg_style, patch_labels == highlight,
                               patch_size, cmap(norm(highlight)))
        if r == 0:
            ax.set_title("clusters" if highlight is None else f"cluster {highlight}",
                         fontsize=title_fontsize)

        if entropy is not None:
            ax = fig.add_subplot(gs[2 * r, len(feats) + 1])
            ent_im = _patch_overlay(ax, img[bg_idx], bg_style, entropy[sl], patch_size,
                                    cmap=_ENTROPY_CMAP, vmin=0.0, vmax=1.0)
            entropy_axes.append(ax)
            if r == 0:
                ax.set_title("entropy", fontsize=title_fontsize)

        if show_map:
            ax = _global_ax(fig, (gs[2 * r, n_col - 1],))
            if row is not None:
                ax.scatter([row["center_lon"]], [row["center_lat"]], s=80, color="red",
                           marker="*", transform=_PROJ, zorder=3)
            if r == 0:
                ax.set_title("location", fontsize=title_fontsize)

        cap = fig.add_subplot(gs[2 * r + 1, :]); cap.axis("off")
        prefix = ("" if highlight is None else
                  f"cluster {highlight}: {int((patch_labels == highlight).sum())} patches     ")
        cap.text(0.01, 0.5, prefix + _format_meta(row, metadata_fields),
                 va="center", ha="left", fontsize=meta_fontsize, family="monospace")

    if entropy_axes:
        fig.colorbar(ent_im, ax=entropy_axes, fraction=0.046, pad=0.02, label="entropy")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


_PROJ = ccrs.PlateCarree()


def _global_ax(fig, subplot, extent=None, coastlines=True, draw_labels=False):
    """Lon/lat cartopy axis with land, coastlines and gridlines.

    subplot : (nrows, ncols, index) passed to fig.add_subplot.
    extent  : (lon_min, lon_max, lat_min, lat_max); None frames the whole globe.
    """
    ax = fig.add_subplot(*subplot, projection=_PROJ)
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=_PROJ)
    if coastlines:
        ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
        ax.coastlines(linewidth=0.5, zorder=1)
    ax.gridlines(draw_labels=draw_labels, linewidth=0.3, alpha=0.5)
    return ax


def _patch_map_data(dataset, labels, patch_size, drop_noise=False):
    """Per-patch (lon, lat, labels, times) for mapping, plus the keep mask.
    Patches with no timestamp (and noise, when drop_noise) are dropped."""
    lon, lat = dataset.get_patch_coords(patch_size)
    ts = dataset.get_patch_times(patch_size)
    labels = _as_int_labels(labels)
    assert labels.size == lon.size, f"{labels.size} labels vs {lon.size} patches"
    keep = ~np.isnat(ts)
    if drop_noise:
        keep &= labels >= 0
    return lon[keep], lat[keep], labels[keep], ts[keep], keep


def _fit_extent(lon, lat, margin=2.0):
    """Extent framing every point, with a margin."""
    return (lon.min() - margin, lon.max() + margin, lat.min() - margin, lat.max() + margin)


def _side_colorbar(fig, ax, mappable, label, slot=0, width=0.018, gap=0.12):
    """Colorbar in its own axis to the right of a map axis.  Stealing space from a
    fixed-aspect GeoAxes (fig.colorbar(ax=...)) shrinks the map and lets the
    colorbar clip its lon/lat labels, so the axis is placed explicitly instead.
    slot counts outward from the map, so several stack without colliding."""
    cax = ax.inset_axes([1.0 + gap * (slot + 1), 0.0, width, 1.0], transform=ax.transAxes)
    return fig.colorbar(mappable, cax=cax, label=label)


def _reserve_side_colorbars(fig, n):
    """Shrink the plotting area so n side colorbars fit inside the figure."""
    fig.subplots_adjust(right=0.92 - 0.10 * n)


def _cluster_colorbar(fig, mappable, ax, labels, slot=0, max_ticks=20):
    """Colorbar for discrete cluster labels, ticked only when few enough to read."""
    cbar = _side_colorbar(fig, ax, mappable, "cluster", slot=slot)
    ticks = np.arange(int(labels.min()), int(labels.max()) + 1)
    if ticks.size <= max_ticks:
        cbar.set_ticks(ticks)
    return cbar


def plot_global_cluster_maps(dataset, labels, *, patch_size=8,
                             cmap=None, alpha=0.5, point_size=8, extent=None,
                             coastlines=True, panel_size=8, drop_noise=False,
                             save_dir=None):
    """One global lon/lat map per timestamp; each patch is a dot colored by its
    cluster label (-1 noise shown grey).  Emits a separate figure per timestamp.

    dataset  : CutoutDataset (provides per-patch lon/lat and timestamps).
    Pass the same ``cmap`` used elsewhere so colors match.

    alpha    : dot opacity (dots may overlap).
    extent   : (lon_min, lon_max, lat_min, lat_max); default auto-fits all points,
               shared across timestamps.  Use (-180, 180, -90, 90) for the globe.
    save_dir : if given, save each figure as clusters_<timestamp>.png there.
    """
    lon, lat, labels, ts, _ = _patch_map_data(dataset, labels, patch_size, drop_noise)

    cmap_d, norm = _cluster_cmap_norm(labels, cmap)
    if extent is None:
        extent = _fit_extent(lon, lat)

    for t in np.unique(ts):
        msk = ts == t
        fig = plt.figure(figsize=(panel_size, panel_size * 0.55))
        ax = _global_ax(fig, (1, 1, 1), extent, coastlines, draw_labels=True)
        _reserve_side_colorbars(fig, 1)
        sc = ax.scatter(lon[msk], lat[msk], c=labels[msk], cmap=cmap_d, norm=norm,
                        s=point_size, alpha=alpha, linewidths=0, transform=_PROJ, zorder=2)
        ax.set_title(str(np.datetime64(t, "s")))
        _cluster_colorbar(fig, sc, ax, labels)
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"clusters_{np.datetime64(t, 's')}.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()


def plot_global_field_cluster_maps(dataset, labels, field="gradb2", *, patch_size=8,
                                   log_grads=True, clusters=None, point_size=14,
                                   cluster_size=None, cluster_alpha=0.9,
                                   extent=None, coastlines=True, panel_size=10,
                                   drop_noise=False, cmap=None, save_dir=None):
    """One global map per timestamp: the sampled field where cutouts were taken,
    with cluster squares drawn over it.

    The field only exists at sampled patch locations, so each patch is a square
    colored by its mean value, with a smaller square in the cluster's own color on
    top -- the field stays visible as a border around each cluster square.  The
    field's color scale is shared across timestamps so the panels are comparable.

    field      : channel drawn underneath (default the buoyancy gradient).
    log_grads  : log10 the gradient-magnitude channels before averaging.
    clusters   : labels to overlay (default all); pass e.g. the ids returned by
                 plot_cluster_size_distribution to overlay only the largest.
    save_dir   : if given, save each figure as <field>_clusters_<timestamp>.png.
    """
    ci, name = _resolve_features([field], dataset.channel_names)[0]
    lon, lat, labels, ts, keep = _patch_map_data(dataset, labels, patch_size, drop_noise)
    values = dataset.get_patch_features(patch_size, log_grads=log_grads)[keep, ci]

    shown = (np.ones(labels.size, bool) if clusters is None
             else np.isin(labels, np.asarray(clusters, dtype=int)))
    if not shown.any():
        raise ValueError(f"none of clusters={clusters} are present")

    f_cmap, vmin, vmax = _field_style(name, values)
    f_label = _axis_label(name, log_grads and name in dataset.log_scaled_channels)
    cmap_d, norm = _cluster_cmap_norm(labels[shown], cmap)
    cluster_size = cluster_size or point_size * 0.35
    if extent is None:
        extent = _fit_extent(lon, lat)

    for t in np.unique(ts):
        msk = ts == t
        sel = msk & shown
        fig = plt.figure(figsize=(panel_size, panel_size * 0.55))
        ax = _global_ax(fig, (1, 1, 1), extent, coastlines, draw_labels=True)
        _reserve_side_colorbars(fig, 2)
        fm = ax.scatter(lon[msk], lat[msk], c=values[msk], cmap=f_cmap, vmin=vmin, vmax=vmax,
                        s=point_size, marker="s", linewidths=0, transform=_PROJ, zorder=2)
        cl = ax.scatter(lon[sel], lat[sel], c=labels[sel], cmap=cmap_d, norm=norm,
                        s=cluster_size, marker="s", alpha=cluster_alpha, linewidths=0,
                        transform=_PROJ, zorder=3)
        ax.set_title(f"{np.datetime64(t, 's')}  |  {int(msk.sum()):,} patches")
        _side_colorbar(fig, ax, fm, f_label, slot=0)
        _cluster_colorbar(fig, cl, ax, labels[shown], slot=1)
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"{name}_clusters_{np.datetime64(t, 's')}.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()


def plot_cluster_size_distribution(labels, *, top_n=None, drop_noise=False, log=False,
                                   cumulative=True, cmap=None, max_tick_labels=40,
                                   figsize=(14, 5)):
    """Share of the data held by each cluster, largest first.

    labels     : flat per-patch cluster labels; -1 (and NaN) noise is a cluster of
                 its own unless drop_noise.
    top_n      : draw only the largest n bars.
    cumulative : overlay the running coverage of the ordered clusters, which is
                 what tells you the top_n worth passing to the other cluster plots.
    log        : log the percentage axis, for long tails of tiny clusters.

    Bars carry the cluster's own color, so they match the map and patch views.
    Returns (cluster_ids, percent) for the bars drawn.
    """
    ids, counts, frac = _cluster_order(labels, drop_noise)
    pct = 100 * frac
    cmap_d, norm = _cluster_cmap_norm(ids, cmap)

    show = ids.size if top_n is None else min(top_n, ids.size)
    x = np.arange(show)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, pct[:show], color=cmap_d(norm(ids[:show])), width=1.0)
    ax.set_xlabel("cluster (descending size)")
    ax.set_ylabel("% of patches")
    if log:
        ax.set_yscale("log")
    if show <= max_tick_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(ids[:show], rotation=90)
    ax.set_title(f"{ids.size} clusters | largest {pct[0]:.1f}% "
                 f"(cluster {ids[0]}) | shown {show} cover {pct[:show].sum():.1f}%")

    if cumulative:
        ax2 = ax.twinx()
        ax2.plot(x, np.cumsum(pct[:show]), color="k", lw=1.5)
        ax2.set_ylabel("cumulative % of patches")
        ax2.set_ylim(0, 100)

    fig.tight_layout()
    plt.show()
    return ids[:show], pct[:show]


def plot_cluster_feature_spread(dataset, labels, features=None, *, patch_size=8,
                                entropy=None, show_time=False, log_grads=True,
                                top_n=None, n_clusters=20, max_per_cluster=2000,
                                kind="violin", preproc=False, drop_noise=False,
                                seed=0, panel_size=3, cmap=None):
    """Spread of one or more features within each cluster: one row per feature,
    one violin (or box) per cluster.

    Every patch contributes its per-channel mean, so a violin is the distribution
    of that feature over the patches the cluster holds.

    features        : channel names/indices (default: all feature channels).
    entropy         : optional per-patch ensemble entropy (NEMI.entropy), added as
                      its own row.
    show_time       : add a row for the distribution of patch timestamps.
    log_grads       : log10 the gradient-magnitude channels (gradb2, gradrho2)
                      before averaging, since they span orders of magnitude.
    top_n           : plot the largest n clusters, size-ordered.  When None,
                      n_clusters are drawn at random from every cluster present,
                      so the small ones are represented too.
    max_per_cluster : patches sampled per cluster before plotting.
    preproc         : plot the training transform instead of physical units.
    """
    if kind not in ("violin", "box"):
        raise ValueError("kind must be 'violin' or 'box'")

    labels = _as_int_labels(labels)
    values = dataset.get_patch_features(patch_size, preproc=preproc,
                                        log_grads=log_grads)           # (N_patches, C)
    assert labels.size == values.shape[0], \
        f"{labels.size} labels vs {values.shape[0]} patches"

    feats = _resolve_features(features, dataset.channel_names)
    logged = dataset.log_scaled_channels if (log_grads or preproc) else []
    panels = [(values[:, ci], _axis_label(name, name in logged, preproc), False)
              for ci, name in feats]
    if show_time:
        panels.append((mdates.date2num(dataset.get_patch_times(patch_size)), "time", True))
    if entropy is not None:
        entropy = np.asarray(entropy)
        assert entropy.size == labels.size, \
            f"{entropy.size} entropy values vs {labels.size} labels"
        panels.append((entropy, "entropy", False))

    ids, counts, pct = _select_clusters(labels, top_n, n_clusters, drop_noise, seed)
    rng = np.random.default_rng(seed)
    members = []
    for c in ids:
        idx = np.flatnonzero(labels == c)
        if idx.size > max_per_cluster:
            idx = rng.choice(idx, size=max_per_cluster, replace=False)
        members.append(idx)

    # non-finite values (NaT timestamps, missing metadata) drop out per panel
    panel_data = []
    for vals, label, is_time in panels:
        data = [vals[idx][np.isfinite(vals[idx])] for idx in members]
        short = [int(c) for c, d in zip(ids, data) if d.size < 2]
        if kind == "violin" and short:
            raise ValueError(f"clusters {short} have <2 finite '{label}' values, which "
                             f"has no KDE; use kind='box'")
        panel_data.append((label, is_time, data))

    cmap_d, norm = _cluster_cmap_norm(ids, cmap)
    colors = cmap_d(norm(ids))
    x = np.arange(ids.size)

    fig, axes = plt.subplots(len(panel_data), 1, sharex=True, squeeze=False,
                             figsize=(max(0.5 * ids.size, 8), panel_size * len(panel_data)))
    for ax, (label, is_time, data) in zip(axes[:, 0], panel_data):
        if kind == "violin":
            parts = ax.violinplot(data, positions=x, widths=0.9,
                                  showextrema=False, showmedians=True)
            for body, col in zip(parts["bodies"], colors):
                body.set_facecolor(col)
                body.set_alpha(0.85)
        else:
            bp = ax.boxplot(data, positions=x, widths=0.7, showfliers=False,
                            patch_artist=True)
            for box, col in zip(bp["boxes"], colors):
                box.set_facecolor(col)
        ax.set_ylabel(label)
        if is_time:
            ax.yaxis_date()
        ax.grid(axis="y", lw=0.3, alpha=0.5)

    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels([f"{c}\n{p:.2f}%" for c, p in zip(ids, pct)], fontsize=8)
    axes[-1, 0].set_xlabel("cluster (label / % of patches)")
    selection = "largest first" if top_n is not None else "random sample"
    fig.suptitle(f"per-patch feature spread by cluster "
                 f"({ids.size} clusters, {selection}, <={max_per_cluster} patches each)")
    fig.tight_layout()
    plt.show()


def _cell_fraction(v):
    """Share of a hexbin cell's patches belonging to the cluster.  Cells holding
    none of it return NaN so they drop out of the map instead of tiling it with
    zeros."""
    f = np.mean(v)
    return f if f > 0 else np.nan


def plot_top_cluster_maps(dataset, labels, *, patch_size=8, top_n=12, n_cols=4,
                          gridsize=60, normalize="count", extent=None, coastlines=True,
                          panel_size=4, drop_noise=False, save_path=None):
    """Where the largest clusters are found: one lon/lat hexbin heat map per
    cluster, pooled over every timestamp (plot_global_cluster_maps instead splits
    by timestamp and shows all clusters at once).

    normalize : 'count'    -> patches of that cluster per cell.
                'fraction' -> that count over all patches in the cell, i.e. the
                              cluster's local prevalence, which divides out the
                              uneven sampling density (see plot_sample_map).
    extent    : (lon_min, lon_max, lat_min, lat_max); default is the whole globe.
    save_path : if given, save the figure there before showing.
    """
    if normalize not in ("count", "fraction"):
        raise ValueError("normalize must be 'count' or 'fraction'")

    labels = _as_int_labels(labels)
    lon, lat = dataset.get_patch_coords(patch_size)
    assert labels.size == lon.size, f"{labels.size} labels vs {lon.size} patches"

    ids, counts, pct = _select_clusters(labels, top_n=top_n, drop_noise=drop_noise)
    hex_kw = dict(gridsize=gridsize, mincnt=1, cmap="magma", transform=_PROJ,
                  extent=extent or (-180, 180, -90, 90), zorder=2)
    bar_label = "patches / cell" if normalize == "count" else "fraction of cell"

    n_rows = -(-ids.size // n_cols)                   # ceil
    fig = plt.figure(figsize=(panel_size * n_cols, panel_size * 0.65 * n_rows))
    for i, (c, n, p) in enumerate(zip(ids, counts, pct)):
        ax = _global_ax(fig, (n_rows, n_cols, i + 1), extent, coastlines)
        msk = labels == c
        if normalize == "count":
            hb = ax.hexbin(lon[msk], lat[msk], **hex_kw)
        else:
            hb = ax.hexbin(lon, lat, C=msk.astype(float),
                           reduce_C_function=_cell_fraction, **hex_kw)
        fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.02, label=bar_label)
        ax.set_title(f"cluster {c} | {n:,} patches | {p:.2f}%", fontsize=panel_size * 3)

    fig.suptitle(f"top {ids.size} clusters by size ({normalize})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def _cutouts_with_cluster(labels, cluster, ppi):
    """(cutout indices, per-cutout patch counts) for the cutouts holding a cluster,
    the cutouts richest in it first."""
    per_cutout = (_as_int_labels(labels) == cluster).reshape(-1, ppi).sum(axis=1)
    idx = np.flatnonzero(per_cutout)
    order = np.argsort(-per_cutout[idx], kind="stable")
    return idx[order], per_cutout[idx][order]


def plot_cluster_cutouts(dataset, labels, cluster, *, features=None, patch_size=8,
                         number_rows=6, start=0, **kwargs):
    """Cutouts where a given cluster appears: the requested fields, an overlay
    tinting just that cluster's patches, a global map locating each cutout, and
    the timestamp in the row caption.

    cluster : the cluster's actual label, as titled by plot_top_cluster_maps and
              returned by plot_cluster_size_distribution -- not a size rank.
    Cutouts are ordered by how many patches of the cluster they hold, so start
    pages down from the richest examples.  Extra keyword arguments go to
    make_image_from_patches (entropy, panel_size, save_path, ...).
    """
    H, W = dataset.X.shape[2], dataset.X.shape[3]
    ppi = (H // patch_size) * (W // patch_size)
    idx, counts = _cutouts_with_cluster(labels, cluster, ppi)
    if idx.size == 0:
        raise ValueError(f"cluster {cluster} is not present in any cutout")
    print(f"cluster {cluster}: {counts.sum():,} patches across {idx.size:,} cutouts "
          f"| richest holds {counts[0]} of {ppi}")
    make_image_from_patches(dataset, labels, features=features, patch_size=patch_size,
                            cutouts=idx, highlight=cluster, show_map=True,
                            number_rows=number_rows, start=start, **kwargs)


def _binned_trend(x, y, nbins):
    """(centers, median) of y binned over x; NaN for empty bins."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    b = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(y[b == i]) if np.any(b == i) else np.nan
                    for i in range(nbins)])
    return centers, med


def field_eda(dataset, field, *, bins=200, log=False, sample_pixels=1_000_000,
              trend_bins=40, panel_size=4, point_size=6, alpha=0.3, rng=None, preprocess=False):
    """Four-panel EDA for one field: pixel PDF, then per-cutout mean vs latitude,
    longitude and time (centres/time come from dataset.metadata, by id).

    dataset : CutoutDataset (raw X, channel_names, ids, metadata).
    field   : channel name, e.g. 'Theta'.
    """
    rng = rng or np.random.default_rng(0)
    ci = dataset.channel_names.index(field)
    if preprocess:
        X = dataset.preprocess_for_training()[:, ci]
    else:
        X = dataset.X[:, ci]                                   # (N, H, W) raw
    label = _field_label(field)

    # pixel PDF (subsampled for speed)
    flat = X.reshape(-1)
    if sample_pixels and flat.size > sample_pixels:
        flat = flat[rng.choice(flat.size, size=sample_pixels, replace=False)]
    flat = flat[np.isfinite(flat)]

    # per-cutout means, and metadata aligned to X rows by id
    means = np.nanmean(X, axis=(1, 2))
    meta = dataset.metadata.reindex(dataset.ids)
    lat = meta["center_lat"].to_numpy(dtype=float)
    lon = meta["center_lon"].to_numpy(dtype=float)
    t   = meta["time_snapshot"].to_numpy()

    fig, axes = plt.subplots(1, 4, figsize=(panel_size * 4, panel_size))

    axes[0].hist(flat, bins=bins, density=True, log=log)
    axes[0].set_xlabel(label); axes[0].set_ylabel("PDF")
    axes[0].set_title(f"{field} pixel PDF  (n={flat.size:,})")

    for ax, v, name in ((axes[1], lat, "latitude"), (axes[2], lon, "longitude")):
        ax.scatter(v, means, s=point_size, alpha=alpha, linewidths=0)
        c, m = _binned_trend(v, means, trend_bins)
        ax.plot(c, m, color="red", lw=2)
        ax.set_xlabel(f"center {name}"); ax.set_ylabel("cutout mean")
        ax.set_title(f"{field} by {name}")

    axes[3].scatter(t, means, s=point_size, alpha=alpha, linewidths=0)
    uniq = np.unique(t[~pd.isna(t)])
    if uniq.size:
        axes[3].plot(uniq, [np.nanmean(means[t == u]) for u in uniq], color="red", lw=2)
    axes[3].set_xlabel("time"); axes[3].set_ylabel("cutout mean")
    axes[3].set_title(f"{field} by time")

    fig.autofmt_xdate()
    fig.suptitle(label)
    fig.tight_layout()
    plt.show()


def channel_correlation(dataset, *, sample_pixels=500_000, annot=True,
                        figsize=(12, 10), rng=None):
    """Channel x channel Pearson correlation over randomly sampled pixels.
    Samples (cutout, row, col) triples so the full stack is never materialized."""
    rng = rng or np.random.default_rng(0)
    N, C, H, W = dataset.X.shape
    n = rng.integers(0, N, sample_pixels)
    h = rng.integers(0, H, sample_pixels)
    w = rng.integers(0, W, sample_pixels)
    corr = pd.DataFrame(dataset.X[n, :, h, w],              # (sample_pixels, C)
                        columns=dataset.channel_names).corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(C)); ax.set_xticklabels(dataset.channel_names, rotation=90)
    ax.set_yticks(range(C)); ax.set_yticklabels(dataset.channel_names)
    if annot:
        for i in range(C):
            for j in range(C):
                ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Pearson r")
    ax.set_title(f"channel correlation  (n={sample_pixels:,} pixels)")
    fig.tight_layout()
    plt.show()
    return corr


def plot_sample_map(dataset, *, density=True, point_size=4, alpha=0.05,
                    gridsize=60, figsize=(14, 5)):
    """Where the cutouts were drawn: scatter of centres plus a hexbin density
    map, to expose geographic sampling bias."""
    meta = dataset.metadata.reindex(dataset.ids)
    lat = meta["center_lat"].to_numpy(dtype=float)
    lon = meta["center_lon"].to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]

    ncol = 2 if density else 1
    fig = plt.figure(figsize=figsize)

    ax = _global_ax(fig, (1, ncol, 1))
    ax.scatter(lon, lat, s=point_size, alpha=alpha, transform=_PROJ, zorder=2)
    ax.set_title(f"cutout centres (n={lat.size:,})")

    if density:
        ax2 = _global_ax(fig, (1, 2, 2))
        hb = ax2.hexbin(lon, lat, gridsize=gridsize, mincnt=1, cmap="magma",
                        transform=_PROJ, extent=(-180, 180, -90, 90))
        fig.colorbar(hb, ax=ax2, fraction=0.025, pad=0.02, label="cutouts / cell")
        ax2.set_title("sampling density")

    fig.tight_layout()
    plt.show()