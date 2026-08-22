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
from einops import rearrange, reduce
from sklearn.preprocessing import QuantileTransformer

__all__ = ["from_patches", "from_cutouts", "from_dataset", "scatter_patches",
           "REQUIRED_CHANNELS", "SCATTER_CHANNELS", "FEATURE_NAMES"]

OMEGA = 7.2921e-5                      # Earth's rotation rate, s^-1

REQUIRED_CHANNELS = ("Theta", "Salt", "gradb2", "relative_vorticity",
                     "strain_mag", "divergence", "coriolis_f")

SCATTER_CHANNELS = ("gradb2", "relative_vorticity", "strain_mag", "divergence")

FEATURE_NAMES = ("Theta_mean", "Salt_mean",
                 "Ro_mean", "Ro_std", "S_f_mean", "S_f_std", "div_f_mean", "div_f_std",
                 "loggradb2_mean", "loggradb2_max", "loggradb2_std")


def _morlet_bank(H, W, J, L):
    """Oriented band-pass filters in Fourier space, J octaves x L orientations.

    Polar-separable Morlet-like wavelets.  Orientations span [0, pi) because the
    fields are real, so theta and theta+pi carry the same information.
    """
    ky, kx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
    kr, kth = np.hypot(ky, kx), np.arctan2(ky, kx)
    bank = np.empty((J, L, H, W), dtype="float32")
    for j in range(J):
        k0 = 0.5 / 2 ** (j + 1)                                  # one octave per j
        radial = np.exp(-((kr - k0) ** 2) / (2 * (k0 / 2.0) ** 2))
        for l in range(L):
            d = np.angle(np.exp(1j * 2 * (kth - l * np.pi / L))) / 2
            bank[j, l] = radial * np.exp(-(d ** 2) / (2 * (np.pi / (2 * L)) ** 2))
    bank[:, :, 0, 0] = 0.0                                       # drop DC
    return bank


def scatter_patches(field, patch_size, J=3, L=4, reduced=True):
    """First-order wavelet scattering of (n, H, W) cutouts, aggregated to patches.

    |x * psi_{j,theta}| is block-averaged over each patch -- the low-pass and
    subsample step of a scattering transform, landing exactly on the get_patches
    grid.  Unlike a patch mean or std this responds to *arrangement*: a filament
    and a blob of equal amplitude give different coefficients.

    reduced collapses the L orientations to (energy, anisotropy) per octave.
    Absolute orientation is arbitrary for a front; how strongly oriented it is is
    not.  reduced=False keeps every orientation instead.
    """
    n, H, W = field.shape
    bank = _morlet_bank(H, W, J, L)
    Fh = np.fft.fft2(field - field.mean(axis=(1, 2), keepdims=True), axes=(1, 2))
    out = {}
    for j in range(J):
        per_o = np.stack([
            reduce(np.abs(np.fft.ifft2(Fh * bank[j, l], axes=(1, 2))),
                   'n (h p1) (w p2) -> (n h w)', 'mean', p1=patch_size, p2=patch_size)
            for l in range(L)])
        if reduced:
            hi, lo = per_o.max(0), per_o.min(0)
            out[f"j{j}_energy"] = per_o.mean(0)
            out[f"j{j}_aniso"] = (hi - lo) / (hi + lo + 1e-30)
        else:
            out.update({f"j{j}_o{l}": per_o[l] for l in range(L)})
    return out


def _moments(field, want):
    """mean / std / max over the pixels of each patch in a (n, p, p) block."""
    out = {"mean": field.mean(axis=(1, 2))}
    if "std" in want:
        out["std"] = np.sqrt(np.maximum((field ** 2).mean(axis=(1, 2)) - out["mean"] ** 2, 0.0))
    if "max" in want:
        out["max"] = field.max(axis=(1, 2))
    return out


def _moment_block(b, ix, gfloor, f_floor):
    """The eleven moment features for one block of (n, C, p, p) patches."""
    f = b[:, ix["coriolis_f"]]
    f_signed = np.where(f < 0, -1.0, 1.0) * np.maximum(np.abs(f), f_floor)
    part = {"Theta_mean": b[:, ix["Theta"]].mean(axis=(1, 2)),
            "Salt_mean": b[:, ix["Salt"]].mean(axis=(1, 2))}
    for name, field in (("Ro", b[:, ix["relative_vorticity"]] / f_signed),
                        ("S_f", b[:, ix["strain_mag"]] / np.abs(f_signed)),
                        ("div_f", b[:, ix["divergence"]] / np.abs(f_signed))):
        m = _moments(field, ("std",))
        part[f"{name}_mean"], part[f"{name}_std"] = m["mean"], m["std"]
    m = _moments(np.log10(np.where(b[:, ix["gradb2"]] > 0, b[:, ix["gradb2"]], gfloor)), ("std", "max"))
    part["loggradb2_mean"], part["loggradb2_std"], part["loggradb2_max"] = \
        m["mean"], m["std"], m["max"]
    return part, (np.abs(f) < f_floor).any(axis=(1, 2))


def _gradb2_floor(patches, index):
    """A positive stand-in for exact-zero gradb2, so it does not log to -inf.
    Substituted only where the value is non-positive, never used to clip real
    values -- it comes from a subsample, so clipping against it would make the
    result depend on which rows happened to be sampled."""
    sample = patches[::max(1, len(patches) // 64), index]
    pos = sample[sample > 0]
    return float(pos.min()) if pos.size else 1e-20


def from_patches(patches, channel_names, *, quantile=False, equator_deg=5.0,
                 block=200_000, seed=0, scattering=False):
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
    if scattering:
        raise ValueError("scattering needs whole cutouts for its wavelet support; "
                         "use from_cutouts or from_dataset")
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
        part, flag = _moment_block(np.asarray(patches[i:i + block], dtype="float32"),
                                   ix, gfloor, f_floor)
        flags.append(flag)
        for k, v in part.items():
            acc.setdefault(k, []).append(v)

    features = pd.DataFrame({k: np.concatenate(acc[k]) for k in FEATURE_NAMES})
    if quantile:
        # TODO return the fitted transformer so a held-out split can reuse it
        qt = QuantileTransformer(output_distribution="normal", subsample=500_000,
                                 random_state=seed)
        features = pd.DataFrame(qt.fit_transform(features), columns=features.columns)
    return features, np.concatenate(flags)


def from_cutouts(X, channel_names, patch_size, *, quantile=False, scattering=True,
                 J=3, L=4, reduced=True, equator_deg=5.0, block=256, seed=0):
    """Features for every patch of (N, C, H, W) **raw** cutouts, in patch order.

    Same eleven moment features as ``from_patches``, plus first-order wavelet
    scattering of the dynamical channels when ``scattering`` is set.  Scattering
    needs whole cutouts rather than isolated patches: the wavelets extend past a
    patch boundary, which is what lets a coefficient describe the patch's texture
    in the context of its surroundings.

    J octaves x L orientations per channel in SCATTER_CHANNELS.  With reduced,
    that is 2*J features per channel (energy and anisotropy per octave); without,
    J*L.  gradb2 is scattered on its log, its raw range spanning decades.

    Returns (features, near_equator), as ``from_patches``.
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"expected (N, C, H, W) cutouts, got shape {X.shape}")
    missing = [c for c in REQUIRED_CHANNELS if c not in channel_names]
    if missing:
        raise ValueError(f"channels required by engineer_features are missing: {missing}. "
                         f"got: {list(channel_names)}")
    if X.shape[1] != len(channel_names):
        raise ValueError(f"cutouts have {X.shape[1]} channels but "
                         f"{len(channel_names)} names were given")
    if X.shape[2] % patch_size or X.shape[3] % patch_size:
        raise ValueError(f"patch_size {patch_size} does not divide {X.shape[2:]}")

    ix = {c: list(channel_names).index(c) for c in REQUIRED_CHANNELS}
    f_floor = 2 * OMEGA * np.sin(np.deg2rad(equator_deg))
    sample = X[::max(1, len(X) // 64), ix["gradb2"]]
    pos = sample[sample > 0]
    gfloor = float(pos.min()) if pos.size else 1e-20

    acc, flags = {}, []
    for i in range(0, len(X), block):
        cut = np.asarray(X[i:i + block], dtype="float32")
        pat = rearrange(cut, 'n c (h p1) (w p2) -> (n h w) c p1 p2',
                        p1=patch_size, p2=patch_size)
        part, flag = _moment_block(pat, ix, gfloor, f_floor)
        flags.append(flag)
        if scattering:
            for ch in SCATTER_CHANNELS:
                fld = cut[:, ix[ch]]
                if ch.startswith("grad"):
                    fld = np.log10(np.where(fld > 0, fld, gfloor))
                for k, v in scatter_patches(fld, patch_size, J, L, reduced).items():
                    part[f"sc_{ch}_{k}"] = v
        for k, v in part.items():
            acc.setdefault(k, []).append(v)

    names = list(FEATURE_NAMES) + [k for k in acc if k.startswith("sc_")]
    features = pd.DataFrame({k: np.concatenate(acc[k]) for k in names})
    if quantile:
        # TODO return the fitted transformer so a held-out split can reuse it
        qt = QuantileTransformer(output_distribution="normal", subsample=500_000,
                                 random_state=seed)
        features = pd.DataFrame(qt.fit_transform(features), columns=features.columns)
    return features, np.concatenate(flags)


def from_dataset(dataset, patch_size, **kwargs):
    """``from_cutouts`` over a CutoutDataset, using its raw cutout images."""
    return from_cutouts(dataset.X, dataset.channel_names, patch_size, **kwargs)
