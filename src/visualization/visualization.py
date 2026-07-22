import colorsys
import itertools
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from einops import rearrange

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


def make_image_from_patches(cutouts_dataloader, patched_loader, labels, *,
                            features=None, label_overlay_on=0,
                            batch_index=0, batch_size=64, patch_size=8,
                            number_rows=6, metadata=None,
                            metadata_fields=_DEFAULT_META_FIELDS,
                            panel_size=3, cmap=None,
                            meta_fontsize=None, title_fontsize=None, save_path=None):
    """Grid of cutouts: one panel per selected feature (titled with its zarr
    channel name, colored with its llc4320 field colormap) + a cluster-label
    overlay, and a metadata caption under each row.

    features         : channel names/indices to show (default: all channels).
    label_overlay_on : channel name/index used as the overlay background.
    metadata         : id-indexed DataFrame; defaults to cutouts_dataloader.dataset.metadata.
                       Looked up per cutout by its id (order-independent).
    save_path        : if given, save the figure there before showing.

    Requires non-shuffled, deterministic loaders so ``labels`` (flat, in loader
    order) line up with the re-iterated patches.
    """
    ds = getattr(cutouts_dataloader, "dataset", None)
    channel_names = getattr(ds, "channel_names", None)
    if metadata is None:
        metadata = getattr(ds, "metadata", None)
    ids = getattr(ds, "ids", None)
    feats = _resolve_features(features, channel_names)
    bg_idx = _resolve_features([label_overlay_on], channel_names)[0][0]
    bg_name = channel_names[bg_idx] if channel_names else None

    meta_fontsize = meta_fontsize or int(panel_size * 6)
    title_fontsize = title_fontsize or int(panel_size * 4)

    batch_imgs = next(itertools.islice(iter(cutouts_dataloader), batch_index, None))
    batch_patches = next(itertools.islice(iter(patched_loader), batch_index, None))
    imgs_in_batch, patches_per_image = batch_patches.shape[0], batch_patches.shape[1]
    number_rows = min(number_rows, imgs_in_batch)

    labels = np.asarray(labels)
    cmap, norm = _cluster_cmap_norm(labels, cmap)   # grey slot for -1 noise

    n_col = len(feats) + 1
    fig = plt.figure(figsize=(panel_size * n_col, panel_size * 1.4 * number_rows))
    gs = fig.add_gridspec(number_rows * 2, n_col,
                          height_ratios=[6, 2] * number_rows, hspace=0.06, wspace=0.05)

    for r in range(number_rows):
        img = np.asarray(batch_imgs[r])
        H, W = img.shape[1], img.shape[2]
        n_h, n_w = H // patch_size, W // patch_size

        for col, (idx, name) in enumerate(feats):
            ax = fig.add_subplot(gs[2 * r, col])
            f_cmap, vmin, vmax = _field_style(name, img[idx])
            ax.imshow(img[idx], cmap=f_cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=title_fontsize)

        ax = fig.add_subplot(gs[2 * r, len(feats)])
        bg = rearrange(np.asarray(batch_patches[r][:, bg_idx]),
                       '(h w) p1 p2 -> (h p1) (w p2)', h=n_h, w=n_w)
        bg_cmap, bg_vmin, bg_vmax = _field_style(bg_name, bg)
        ax.imshow(bg, extent=[0, W, H, 0], cmap=bg_cmap, vmin=bg_vmin, vmax=bg_vmax)
        start = batch_index * batch_size * patches_per_image + r * patches_per_image
        patch_labels = labels[start:start + patches_per_image]
        assert patch_labels.size == n_h * n_w, \
            f"expected {n_h * n_w} labels, got {patch_labels.size}"
        ax.imshow(patch_labels.reshape(n_h, n_w), cmap=cmap, norm=norm, alpha=0.9,
                  extent=[0, W, H, 0], interpolation="nearest")
        for g in range(0, H + 1, patch_size):
            ax.axhline(g, color="k", lw=0.8, alpha=0.6)
        for g in range(0, W + 1, patch_size):
            ax.axvline(g, color="k", lw=0.8, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("clusters", fontsize=title_fontsize)

        cap = fig.add_subplot(gs[2 * r + 1, :]); cap.axis("off")
        k = batch_index * batch_size + r
        row = (metadata.loc[ids[k]]
               if metadata is not None and ids is not None and ids[k] in metadata.index
               else None)
        cap.text(0.01, 0.5, _format_meta(row, metadata_fields),
                 va="center", ha="left", fontsize=meta_fontsize, family="monospace")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_global_cluster_maps(cutouts_dataloader, patched_loader, labels, *,
                             cmap=None, alpha=0.5, point_size=8, extent=None,
                             coastlines=True, panel_size=8, drop_noise=False,
                             save_dir=None):
    """One global lon/lat map per timestamp; each patch is a dot colored by its
    cluster label (-1 noise shown grey).  Emits a separate figure per timestamp.

    XC/YC (lon/lat) come from the patch coordinate channels (passed through
    unnormalized by the loader); each patch's timestamp is its cutout's
    ``time_snapshot``.  Pass the same ``cmap`` used elsewhere so colors match.

    alpha    : dot opacity (dots may overlap).
    extent   : (lon_min, lon_max, lat_min, lat_max); default auto-fits all points,
               shared across timestamps.  Use (-180, 180, -90, 90) for the globe.
    save_dir : if given, save each figure as clusters_<timestamp>.png there.
    """
    ds = getattr(cutouts_dataloader, "dataset", None)
    channel_names, metadata, ids = ds.channel_names, ds.metadata, ds.ids
    xc_i, yc_i = channel_names.index("XC"), channel_names.index("YC")

    lons, lats, ppi = [], [], None
    for batch in patched_loader:                       # (B, ppi, C, p, p)
        ppi = batch.shape[1]
        lons.append(np.asarray(batch[:, :, xc_i].mean(dim=(-1, -2))).reshape(-1))
        lats.append(np.asarray(batch[:, :, yc_i].mean(dim=(-1, -2))).reshape(-1))
    lon, lat = np.concatenate(lons), np.concatenate(lats)

    labels = np.asarray(labels)
    assert labels.size == lon.size, f"{labels.size} labels vs {lon.size} patches"
    assert len(ids) * ppi == lon.size, "cutout ids and patches are misaligned"

    ts = np.repeat(metadata["time_snapshot"].reindex(np.asarray(ids)).values, ppi)
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