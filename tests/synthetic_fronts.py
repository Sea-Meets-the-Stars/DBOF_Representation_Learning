"""A real FrontStore on disk, small enough to build per test.

Writing a genuine store rather than stubbing the reader keeps the loader
honest about the store's actual shapes: int32 ids, float32 statistics, the
NaN curvature columns, and the cross table's separate schema.
"""
import numpy as np
import pandas as pd

from front_finding.store import FrontStore

DATE = "20111204_000000"
DATE2 = "20111206_180000"


def _geometry(n, seed=0):
    rng = np.random.default_rng(seed)
    curv = rng.normal(size=n)
    curv[: n // 3] = np.nan            # the store really does this
    return pd.DataFrame({
        "label": np.arange(1, n + 1, dtype="int64"),
        "name": [f"20111204TT000000_{i}.0N_2.0E" for i in range(n)],
        "time": ["2011-12-04 00:00:00"] * n,
        "npix": rng.integers(7, 400, n).astype("int64"),
        "y0": np.zeros(n, "int64"), "y1": np.full(n, 8, "int64"),
        "x0": np.zeros(n, "int64"), "x1": np.full(n, 8, "int64"),
        "centroid_lat": rng.uniform(-70, 70, n),
        "centroid_lon": rng.uniform(-180, 180, n),
        "length_km": rng.uniform(5, 500, n),
        "orientation": rng.uniform(0, 180, n),
        "num_branches": rng.integers(0, 9, n).astype("int64"),
        "mean_curvature": curv,
        "curvature_direction": curv,
    })


def _properties(n, seed=0, prefix=""):
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame({
        "flabel": np.arange(1, n + 1, dtype="int64"),
        "npix": rng.integers(7, 400, n).astype("int64"),
        f"{prefix}gradb2_mean": rng.lognormal(-30, 3, n),   # spans decades
        f"{prefix}gradb2_std": rng.lognormal(-30, 3, n),
        f"{prefix}turner_angle_mean": rng.uniform(-90, 90, n),
        f"{prefix}density_mean": rng.uniform(20, 30, n),
        # Kinematic fields and the divisor, so the div_by_f path is reachable.
        f"{prefix}coriolis_f_mean": np.r_[
            rng.uniform(-1.4e-4, -2e-5, n // 2),
            rng.uniform(2e-5, 1.4e-4, n - n // 2)],
        f"{prefix}divergence_mean": rng.normal(0, 1e-5, n),
        f"{prefix}divergence_std": rng.uniform(1e-6, 3e-5, n),
        f"{prefix}divergence_skew": rng.normal(0, 1, n),
        f"{prefix}divergence_min": rng.normal(-3e-5, 1e-5, n),
        f"{prefix}relative_vorticity_mean": rng.normal(0, 1e-5, n),
        f"{prefix}relative_vorticity_std": rng.uniform(1e-6, 3e-5, n),
    })


def build_store(path, n=40, dates=(DATE,), cross=True):
    """A store with *n* fronts per snapshot, find/group/colocate all done."""
    store = FrontStore.open(str(path), mode="w")
    store.set_build_attrs(build_version="TEST", pipeline="SURF", run_id="t")
    binary = np.zeros((16, 16), dtype=bool)
    binary[4:12, 8] = True
    labels = (binary * 1).astype(np.int32)
    for date in dates:
        store.write_binary(date, binary)
        store.write_group(date, labels, _geometry(n))
        store.write_properties(
            date, _properties(n),
            cross_properties=_properties(n, seed=5) if cross else None)
    return store
