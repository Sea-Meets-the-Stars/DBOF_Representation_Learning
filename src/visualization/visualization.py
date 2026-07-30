import colorsys
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
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
    for g in range(0, H + 1, patch_size):
        ax.axhline(g, color="k", lw=0.8, alpha=0.6)
    for g in range(0, W + 1, patch_size):
        ax.axvline(g, color="k", lw=0.8, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def make_image_from_patches(dataset, labels, *, entropy=None,
                            features=None, label_overlay_on=0,
                            patch_size=8, start=0, number_rows=6,
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
    start            : index of the first cutout to show.
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

    number_rows = min(number_rows, len(imgs) - start)
    H, W = imgs.shape[2], imgs.shape[3]
    n_h, n_w = H // patch_size, W // patch_size
    ppi = n_h * n_w

    labels = np.asarray(labels)
    cmap, norm = _cluster_cmap_norm(labels, cmap)   # grey slot for -1 noise

    if entropy is not None:
        entropy = np.asarray(entropy)
        assert entropy.size == labels.size, \
            f"{entropy.size} entropy values vs {labels.size} labels"

    n_col = len(feats) + 1 + (1 if entropy is not None else 0)
    fig = plt.figure(figsize=(panel_size * n_col, panel_size * 1.4 * number_rows))
    gs = fig.add_gridspec(number_rows * 2, n_col,
                          height_ratios=[6, 2] * number_rows, hspace=0.06, wspace=0.05)
    entropy_axes = []

    for r in range(number_rows):
        c = start + r
        img = np.asarray(imgs[c])
        sl = slice(c * ppi, (c + 1) * ppi)

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
        _patch_overlay(ax, img[bg_idx], bg_style, patch_labels, patch_size,
                       cmap=cmap, norm=norm)
        if r == 0:
            ax.set_title("clusters", fontsize=title_fontsize)

        if entropy is not None:
            ax = fig.add_subplot(gs[2 * r, len(feats) + 1])
            ent_im = _patch_overlay(ax, img[bg_idx], bg_style, entropy[sl], patch_size,
                                    cmap=_ENTROPY_CMAP, vmin=0.0, vmax=1.0)
            entropy_axes.append(ax)
            if r == 0:
                ax.set_title("entropy", fontsize=title_fontsize)

        cap = fig.add_subplot(gs[2 * r + 1, :]); cap.axis("off")
        row = metadata.loc[ids[c]] if ids[c] in metadata.index else None
        cap.text(0.01, 0.5, _format_meta(row, metadata_fields),
                 va="center", ha="left", fontsize=meta_fontsize, family="monospace")

    if entropy_axes:
        fig.colorbar(ent_im, ax=entropy_axes, fraction=0.046, pad=0.02, label="entropy")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


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
    lon, lat = dataset.get_patch_coords(patch_size)
    ts = dataset.get_patch_times(patch_size)

    labels = np.asarray(labels)
    assert labels.size == lon.size, f"{labels.size} labels vs {lon.size} patches"

    keep = ~np.isnat(ts)
    if drop_noise:
        keep &= labels >= 0
    lon, lat, labels, ts = lon[keep], lat[keep], labels[keep], ts[keep]

    cmap_d, norm = _cluster_cmap_norm(labels, cmap)
    if extent is None:
        m = 2.0
        extent = (lon.min() - m, lon.max() + m, lat.min() - m, lat.max() + m)
    ticks = np.arange(int(labels.min()), int(labels.max()) + 1)
    proj = ccrs.PlateCarree()

    for t in np.unique(ts):
        msk = ts == t
        fig = plt.figure(figsize=(panel_size, panel_size * 0.55))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(extent, crs=proj)
        if coastlines:
            ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
            ax.coastlines(linewidth=0.5, zorder=1)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        sc = ax.scatter(lon[msk], lat[msk], c=labels[msk], cmap=cmap_d, norm=norm,
                        s=point_size, alpha=alpha, linewidths=0, transform=proj, zorder=2)
        ax.set_title(str(np.datetime64(t, "s")))
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, label="cluster")
        if ticks.size <= 20:
            cbar.set_ticks(ticks)
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"clusters_{np.datetime64(t, 's')}.png"),
                        dpi=150, bbox_inches="tight")
        plt.show()


def _field_label(name):
    """Physical label from the llc4320 registry, falling back to the raw name."""
    spec = _FIELD_CMAPS.get(name)
    return spec[1] if spec else name


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

    proj = ccrs.PlateCarree()
    ncol = 2 if density else 1
    fig, axes = plt.subplots(1, ncol, figsize=figsize, squeeze=False,
                             subplot_kw={"projection": proj})

    ax = axes[0][0]
    ax.set_global(); ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
    ax.coastlines(linewidth=0.5, zorder=1)
    ax.scatter(lon, lat, s=point_size, alpha=alpha, transform=proj, zorder=2)
    ax.set_title(f"cutout centres (n={lat.size:,})")

    if density:
        ax2 = axes[0][1]
        ax2.set_global(); ax2.coastlines(linewidth=0.5)
        hb = ax2.hexbin(lon, lat, gridsize=gridsize, mincnt=1, cmap="magma",
                        transform=proj, extent=(-180, 180, -90, 90))
        fig.colorbar(hb, ax=ax2, fraction=0.025, pad=0.02, label="cutouts / cell")
        ax2.set_title("sampling density")

    fig.tight_layout()
    plt.show()