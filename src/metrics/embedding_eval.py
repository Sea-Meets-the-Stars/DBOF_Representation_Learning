"""Quick validation that patch embeddings group semantically similar patches.

Given per-patch embeddings (from any encoder) and a CutoutDataset, checks whether
nearest neighbors in embedding space are more similar than random -- by physical
properties (per-channel means) and by geography (lon/lat) -- and renders query +
nearest-neighbor patch tiles.  Aligned with get_patches order (patch_size=8).
"""
import numpy as np
import matplotlib.pyplot as plt
from cuml.neighbors import NearestNeighbors

import visualization.visualization as vis


def nearest_neighbors(embeddings, k=10):
    """k nearest neighbors per row, self excluded (cuML GPU)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    dist, idx = nn.kneighbors(embeddings)
    return np.asarray(dist)[:, 1:], np.asarray(idx)[:, 1:]


def patch_descriptors(dataset, patch_size):
    """Per-patch physical descriptors (raw per-channel means) + (lon, lat), in get_patches order."""
    desc = dataset.get_patch_features(patch_size)                                        # (Np, C)
    lon, lat = dataset.get_patch_coords(patch_size)
    return desc, lon, lat, dataset.channel_names


def neighbor_consistency(embeddings, descriptors, channel_names, k=10, seed=0):
    """Per-channel |Δdescriptor| between a patch and its k-NN vs a random baseline.
    ratio < 1 => neighbors are physically more similar than random."""
    z = (descriptors - descriptors.mean(0)) / (descriptors.std(0) + 1e-8)
    _, idx = nearest_neighbors(embeddings, k)
    nn_diff = np.abs(z[:, None, :] - z[idx]).mean(axis=(0, 1))            # (C,)
    rand = np.random.default_rng(seed).integers(0, len(z), size=idx.shape)
    rand_diff = np.abs(z[:, None, :] - z[rand]).mean(axis=(0, 1))         # (C,)
    ratio = nn_diff / rand_diff

    print(f"{'channel':<20} {'nn':>8} {'random':>8} {'ratio':>8}")
    for c, name in enumerate(channel_names):
        print(f"{name:<20} {nn_diff[c]:>8.3f} {rand_diff[c]:>8.3f} {ratio[c]:>8.3f}")
    print(f"{'OVERALL':<20} {nn_diff.mean():>8.3f} {rand_diff.mean():>8.3f} {ratio.mean():>8.3f}")
    print("(ratio < 1 => neighbors more physically similar than random)")
    return {"per_channel": dict(zip(channel_names, ratio)), "overall": float(ratio.mean())}


def _haversine_km(lon1, lat1, lon2, lat2):
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(d))


def neighbor_geographic(embeddings, lon, lat, k=10, seed=0):
    """Median great-circle distance to k-NN vs random baseline.
    ratio < 1 => neighbors are geographically closer than random."""
    _, idx = nearest_neighbors(embeddings, k)
    nn_km = _haversine_km(lon[:, None], lat[:, None], lon[idx], lat[idx]).mean(axis=1)
    rand = np.random.default_rng(seed).integers(0, len(lon), size=idx.shape)
    rand_km = _haversine_km(lon[:, None], lat[:, None], lon[rand], lat[rand]).mean(axis=1)
    out = {"nn_median_km": float(np.median(nn_km)),
           "random_median_km": float(np.median(rand_km)),
           "ratio": float(np.median(nn_km) / np.median(rand_km))}
    print(f"NN median dist {out['nn_median_km']:.0f} km | random {out['random_median_km']:.0f} km "
          f"| ratio {out['ratio']:.3f}")
    print("(ratio < 1 => neighbors geographically closer than random)")
    return out


def plot_neighbor_examples(dataset, embeddings, patch_size, query_idx, k=8,
                           channel=None, panel_size=1.4):
    """For each query patch, show it + its k nearest neighbors (one channel's field),
    annotated with lat/lon, so you can eyeball whether neighbors look alike."""
    patches = dataset.get_patches(patch_size=patch_size, flatten=False, preproc=False)
    lon, lat = dataset.get_patch_coords(patch_size)
    ci = dataset.channel_names.index(channel) if channel else 0
    name = dataset.channel_names[ci]
    _, idx = nearest_neighbors(embeddings, k)

    ncol = k + 1
    fig, axes = plt.subplots(len(query_idx), ncol,
                             figsize=(panel_size * ncol, panel_size * len(query_idx)), squeeze=False)
    for r, q in enumerate(query_idx):
        for c, m in enumerate([q] + list(idx[q])):
            ax = axes[r][c]
            cmap, vmin, vmax = vis._field_style(name, patches[m, ci])
            ax.imshow(patches[m, ci], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(("query" if c == 0 else f"nn{c}") + f"\n{lat[m]:.1f},{lon[m]:.1f}",
                         fontsize=panel_size * 5)
    fig.suptitle(f"nearest neighbors by embedding (field: {name})")
    fig.tight_layout()
    plt.show()
