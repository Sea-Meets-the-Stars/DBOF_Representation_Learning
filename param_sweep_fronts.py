import numpy as np
import pandas as pd
from nemi_sweep.sweep import run_sweep
import json
import fsspec

from fronts_dataloader.fronts_dataset import FrontDataSource, RawFronts

# load fronts
STORE = ("s3://dbof/globals_for_cutouts/v2_2_03/Fronts/V5/SURF/fronts.zarr")
S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
source = FrontDataSource(STORE, storage_options={"endpoint_url": S3_ENDPOINT})

print(f"{len(source.dates)} snapshots")


FEATURES_LOAD = ["num_branches", "mean_curvature", "orientation",
            "relative_vorticity", "turner_angle", "strain_n", "strain_s", "strain_mag", "divergence",
            "okubo_weiss",
            "gradtheta2", "gradsalt2", "gradb2", "gradeta2",
            "SIarea_max"]


raw = RawFronts.load(source,
                     dates=["20111216_030000"], # TODO remove for full run -> None,
                     features=FEATURES_LOAD)
print(f"{len(raw):,} fronts over {len(source.dates)} snapshots")

# Remove ICE
ICE_COLUMN, ICE_MAX = "SIarea_max", 0.0
ice = raw.table[ICE_COLUMN].to_numpy(float)
clean = ~(ice > ICE_MAX)                       # NaN counts as ice-free
raw = RawFronts(raw.table[clean].reset_index(drop=True),
                raw.groups,
                {**raw.source_info, "n_fronts": int(clean.sum()),
                 "ice_filter": f"{ICE_COLUMN} <= {ICE_MAX}"})

print(f"dropped {(~clean).sum():,} fronts touching ice "
      f"({100*(~clean).mean():.1f}%), {len(raw):,} remain")


#Features for training
FEATURES = ["num_branches", "mean_curvature", "orientation",
            "relative_vorticity", "turner_angle", "strain_n", "strain_s", "strain_mag", "divergence",
            "okubo_weiss",
            "gradtheta2", "gradsalt2", "gradb2", "gradeta2"]

ds = raw.select(FEATURES, stats=("mean", "std", "skew"), scaling="standardize", nan_policy="drop_rows",
                div_by_f=False, ihs_channels=True, log_channels=True, fill_value=0, missing_indicator=True)
print(ds.summary())

EMBEDDING_GRID = {"n_neighbors": [40], "min_dist": [0.0]}
result = run_sweep(ds.X, EMBEDDING_GRID, #CLUSTERING_GRID,
                   n_components=3, device="cpu",
                   fit_size=None, metric_sample=1_000, keep_size=1_000)

print(result.summary())

# SAVE RESULTS TO S3

OUT = "s3://dbof/nemi_results/front_finding/sweeps/embedding_only_v0"
SO = {"endpoint_url": S3_ENDPOINT}

result.table.to_parquet(f"{OUT}/metrics.parquet", storage_options=SO)

with fsspec.open(f"{OUT}/arrays.npz", "wb", **SO) as fh:
    np.savez_compressed(
        fh, row_index=result.row_index,
        **{f"embedding_{i:03d}": e for i, e in enumerate(result.embeddings)},
        **{f"labels_{i:03d}": l for i, l in enumerate(result.labels)})

with fsspec.open(f"{OUT}/manifest.json", "w", **SO) as fh:
    json.dump({"store": STORE, "load_features": FEATURES_LOAD,
               "embedding_grid": EMBEDDING_GRID, "n_fronts": len(raw),
               "n_fitted": result.n_fitted, "failures": result.failures,
               **raw.source_info}, fh, indent=2, default=str)
