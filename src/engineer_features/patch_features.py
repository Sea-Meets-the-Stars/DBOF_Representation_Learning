"""Physical per-patch features for clustering, in place of the raw pixel vector.

A raw patch is ``C * p * p`` numbers whose weight in a distance is set by the pixel
count rather than by physics, and whose kinematic channels carry latitude through
their units.  This module reduces each patch to eleven quantities instead:

- **state** -- Theta and Salt patch means
- **dynamical regime** -- Ro = zeta/f, S/f and div/f, each as a patch mean and std
- **frontal sharpness** -- log10 gradb2 patch mean, max and std

The kinematic terms are divided by f because they share its units; without that a
value partly encodes latitude, f varying by an order of magnitude from the tropics
to 60 degrees.  Ro divides by signed f so cyclonic is positive in both hemispheres;
S/f and div/f divide by |f| so strain stays positive and convergence stays
convergence south of the equator.

Patch order is preserved throughout -- no row is dropped or reordered, so the output
stays aligned with ``get_patches``, ``get_patch_coords`` and any cluster labels.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

__all__ = ["from_patches", "from_dataset", "REQUIRED_CHANNELS", "FEATURE_NAMES"]

OMEGA = 7.2921e-5                      # Earth's rotation rate, s^-1

REQUIRED_CHANNELS = ("Theta", "Salt", "gradb2", "relative_vorticity",
                     "strain_mag", "divergence", "coriolis_f")

FEATURE_NAMES = ("Theta_mean", "Salt_mean",
                 "Ro_mean", "Ro_std", "S_f_mean", "S_f_std", "div_f_mean", "div_f_std",
                 "loggradb2_mean", "loggradb2_max", "loggradb2_std")


def _moments(field, want):
    """mean / std / max over the pixels of each patch in a (n, p, p) block."""
    out = {"mean": field.mean(axis=(1, 2))}
    if "std" in want:
        out["std"] = np.sqrt(np.maximum((field ** 2).mean(axis=(1, 2)) - out["mean"] ** 2, 0.0))
    if "max" in want:
        out["max"] = field.max(axis=(1, 2))
    return out


def _gradb2_floor(patches, index):
    """Smallest positive gradb2 in the data, so exact zeros do not log to -inf.
    Mirrors the loader's _safe_log10."""
    sample = patches[::max(1, len(patches) // 64), index]
    pos = sample[sample > 0]
    return float(pos.min()) if pos.size else 1e-20


def from_patches(patches, channel_names, *, quantile=False, equator_deg=5.0,
                 block=200_000, seed=0):
    """Engineered features for every patch, in the order the patches arrive.

    Args:
        patches: (N_patches, C, p, p) **raw** patches -- ``get_patches(...,
            flatten=False, preproc=False)``.  The training transform must not have
            been applied: Ro needs zeta and f in physical units.
        channel_names: channel name per index of axis 1.
        quantile: map every feature's marginal to a normal, leaving the rank order
            untouched.  The /f ratios are heavy-tailed by construction, and squared
            distance lets a fat tail dominate the placement of extreme patches.
        equator_deg: |f| is floored at its value here, so the /f ratios stay finite
            at the equator.  Patches inside the band are kept and flagged.

    Returns:
        (features, near_equator): a (N_patches, 11) DataFrame and a (N_patches,)
        bool array.  Flagged patches have Ro compressed toward zero by the floor
        rather than blown up, so they can form a spurious quiet group -- check the
        flag against any cluster that looks suspicious.

    Raises:
        ValueError: if a required channel is absent, or patches is not 4-D.
    """
    patches = np.asarray(patches)
    if patches.ndim != 4:
        raise ValueError(f"expected (N_patches, C, p, p) patches, got shape {patches.shape}")
    missing = [c for c in REQUIRED_CHANNELS if c not in channel_names]
    if missing:
        raise ValueError(f"channels required by engineer_features are missing: {missing}. "
                         f"got: {list(channel_names)}")
    if patches.shape[1] != len(channel_names):
        raise ValueError(f"patches has {patches.shape[1]} channels but "
                         f"{len(channel_names)} names were given")

    ix = {c: list(channel_names).index(c) for c in REQUIRED_CHANNELS}
    f_floor = 2 * OMEGA * np.sin(np.deg2rad(equator_deg))
    gfloor = _gradb2_floor(patches, ix["gradb2"])

    acc, flags = {}, []
    for i in range(0, len(patches), block):
        b = np.asarray(patches[i:i + block], dtype="float32")
        f = b[:, ix["coriolis_f"]]
        f_signed = np.where(f < 0, -1.0, 1.0) * np.maximum(np.abs(f), f_floor)
        flags.append((np.abs(f) < f_floor).any(axis=(1, 2)))

        part = {"Theta_mean": b[:, ix["Theta"]].mean(axis=(1, 2)),
                "Salt_mean": b[:, ix["Salt"]].mean(axis=(1, 2))}
        for name, field in (("Ro", b[:, ix["relative_vorticity"]] / f_signed),
                            ("S_f", b[:, ix["strain_mag"]] / np.abs(f_signed)),
                            ("div_f", b[:, ix["divergence"]] / np.abs(f_signed))):
            m = _moments(field, ("std",))
            part[f"{name}_mean"], part[f"{name}_std"] = m["mean"], m["std"]
        m = _moments(np.log10(np.maximum(b[:, ix["gradb2"]], gfloor)), ("std", "max"))
        part["loggradb2_mean"], part["loggradb2_std"], part["loggradb2_max"] = \
            m["mean"], m["std"], m["max"]

        for k, v in part.items():
            acc.setdefault(k, []).append(v)

    features = pd.DataFrame({k: np.concatenate(acc[k]) for k in FEATURE_NAMES})
    if quantile:
        # TODO return the fitted transformer so a held-out split can reuse it
        qt = QuantileTransformer(output_distribution="normal", subsample=500_000,
                                 random_state=seed)
        features = pd.DataFrame(qt.fit_transform(features), columns=features.columns)
    return features, np.concatenate(flags)


def from_dataset(dataset, patch_size, **kwargs):
    """``from_patches`` over a CutoutDataset, pulling raw patches at ``patch_size``."""
    patches = dataset.get_patches(patch_size, flatten=False, preproc=False)
    return from_patches(patches, dataset.channel_names, **kwargs)
