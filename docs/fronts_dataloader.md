# fronts_dataloader

Turns a front store into the matrix NEMI clusters. One row per front, one
column per feature.

```python
from fronts_dataloader.fronts_dataset import FrontDataSource, RawFronts

source = FrontDataSource("s3://dbof/.../fronts.zarr")
source.print_available_features()

raw = RawFronts.load(source)                  # every snapshot in the store
ds  = raw.select(["geometry", "gradb2", "turner_angle"])

nemi.run(ds.X, n=30, output="nemi_out.npz")
```

The cutout path builds vectors out of images. A front is already a row of
numbers, so this is selection and scaling — there is no embedding step.

---

## Selecting features

`select` takes a list that may mix three granularities, in any order:

| you write | you get |
|---|---|
| `"geometry"` | every geometry column — `length_km`, `orientation`, `npix`, … |
| `"properties"` | every `{channel}_{stat}` from the front's own band |
| `"cross"` | every `cross_{channel}_{stat}` from the surrounding band |
| `"gradb2"` | every statistic of that channel: `gradb2_mean`, `gradb2_std`, … |
| `"gradb2_mean"` | that one column |

Order is preserved and duplicates collapse, keeping first position. A channel
does **not** pull in its cross twin: `"gradb2"` gives you the front's own
statistics, `"cross_gradb2"` the surroundings.

### Narrowing to particular statistics

`stats=` limits what a group or channel expands to, so a feature list can name
channels rather than columns:

```python
FEATURES = ["length_km", "orientation", "gradb2", "Theta", "rossby_number"]
ds = raw.select(FEATURES, stats=("mean", "std", "skew"))
```

Two things it deliberately does not touch. A **geometry column** is not a
statistic of anything, so `length_km` and `orientation` survive any `stats`
value. A column **named outright** is exempt too — writing `gradb2_p90` while
`stats=("mean",)` is set still gives you `gradb2_p90`, because naming it is
already the choice.

Asking a channel for a statistic it lacks raises and reports what it has.

Which columns exist depends on how the store was built — `properties_stats`
decides which statistics are there, and `cross` is absent unless the run set a
cross-front radius. `print_available_features()` reports what this store has.

### Columns that are never features

- **`label`, `name`, `time`, `date`** identify a front. They are carried on
  `ds.ids`, parallel to `ds.X`, so a cluster can be joined back to the store.
- **`y0`, `y1`, `x0`, `x1`** are the bounding box on the model grid, an
  artefact of the raster; `centroid_lat` / `centroid_lon` says where the front
  is in physical units.
- **`npix_prop`** is the properties table's copy of `npix`, identical to
  geometry's.

---

## Scaling

`scaling="standardize"` (default) gives each column mean 0, sd 1.
`scaling="normalize"` puts each column in [0, 1]. `scaling=None` leaves
physical values.

Columns are scaled independently, so a feature's units stop mattering — which
they otherwise would: `length_km` reaches ~2000 while `gradb2_mean` is ~1e-13,
and an unscaled distance metric would see only the former.

`log_channels=True` (default) takes log10 of any channel whose name starts with
`grad`, before scaling, seeing through the `cross_` prefix. Those span six
decades on real data. Non-positive values become NaN rather than `-inf`, so the
NaN policy sees them instead of one bad pixel ruining a column.

A constant column keeps its centred value rather than dividing by zero.

---

## Dividing by the Coriolis parameter

`div_by_f=True` divides the kinematic fields by `|f|`, turning them into
Rossby-style ratios so one value means the same dynamics at every latitude.
Left off, those fields carry f's latitude dependence and latitude becomes an
embedding coordinate.

Which fields is decided by `field_scaling.DIV_ABS`, shared with
`llc_cutout_dataloader` so both paths scale the same set: `strain_n`,
`strain_s`, `strain_mag`, `divergence`.

### Only the first two moments

`SCALED_STATS = ("mean", "std")`. The others are left alone, for reasons that
are not arbitrary:

- **`std`** divides by `|f|`, never signed — a negative divisor would make a
  standard deviation negative.
- **`skew`** is scale-invariant: `skew(x/c) == skew(x)` for any `c > 0`, so
  dividing would change nothing at all.
- **`min`, `max`, `p25`, `p75`, `p90`** are order statistics. Under a signed
  divisor they swap end for end, and `p90` would need a `p10` the store does
  not carry.

### relative_vorticity passes through, with a warning

`DIV_SIGNED` fields need a *signed* f, and that cannot be applied to a stored
statistic: `mean(x)/mean(f)` is not `mean(x/f)`, and the two part company
wherever f varies across a front's band.

They are left in raw units rather than refused — `div_by_f` is one flag for the
whole selection, so erroring would block a selection that is otherwise fine.
Instead a `UserWarning` names the columns, and `ds.dropped["unscaled_signed"]`
records them. Everything else in the same call still scales.

Those columns still carry f's latitude dependence, so latitude can leak into
the embedding. The store already has the per-pixel quotient:
**`relative_vorticity` → `rossby_number`**, exact and already signed so
cyclonic is positive in both hemispheres. Select that instead.

### How exact the rest is

The same approximation applies to the `DIV_ABS` fields, which have no stored
normalised counterpart. Measured against `rossby_number` on the SMALL_DATASET
snapshot, for the 96.3% of fronts outside the equator floor:

| | `mean` | `std` |
|---|---|---|
| median | 0.04% | 0.03% |
| p90 | 0.54% | 0.41% |
| p99 | 4.62% | 4.18% |

as a fraction of the quantity's typical magnitude. The driver is how much f
varies within one band — median 0.10%, p90 0.79%, p99 6.7%. The tail is long
meridional fronts and anything straddling the equator. After standardisation
this is unlikely to matter for clustering, but it is an approximation, not an
identity.

The divisor is each front's own `coriolis_f_mean`, sampled by co-location over
the same band as everything else. There is no fallback — a store whose
properties table lacks the channel raises, rather than deriving f from the
centroid and silently describing a different band. Add `coriolis_f` to the
colocate channels and re-run that step.

`equator_deg` (default 5.0) floors `|f|` at its value there; f goes to zero at
the equator and these fields would otherwise divide by ~0.

`ds.dropped["scaled_by_f"]` lists the columns that were divided.

---

## Missing values

`nan_policy` decides what happens when a selected column has NaN:

- **`error`** (default) — refuse, naming the columns, their counts, and what
  the other policies would cost on this selection.
- **`fill`** — substitute `fill_value`, keeping every front and every feature.
- **`drop_rows`** — drop every front with a NaN in any selected column.
- **`drop_columns`** — drop the offending columns, keep every front.

The default refuses because none of the others announces itself except in
`summary()`, and which one is right depends on what you selected. Once chosen,
`ds.dropped` and `ds.summary()` report what went.

Every policy leaves `X` rectangular — `drop_columns` narrows every row alike,
so UMAP always sees a single dimensionality. Selecting `geometry` on the V5
SURF test store gives:

| policy | shape |
|---|---|
| `fill` | `(111654, 14)` — 12 features plus 2 indicators |
| `fill`, `missing_indicator=False` | `(111654, 12)` |
| `drop_columns` | `(111654, 10)` |
| `drop_rows` | `(90580, 12)` |

### Filling

`fill_value` takes `"mean"` (default — the column's own mean over the values it
has) or any number. `missing_indicator=True` (default) adds a
`{column}_missing` binary column beside each filled one.

The choice matters where the fill value carries meaning. On the test store:

| `fill_value` | value used | z of filled fronts | z of measured |
|---|---|---|---|
| `"mean"` | 0.0997 | +0.000 | −0.000 |
| `0.0` | 0.0000 | −0.638 | +0.149 |

`mean_curvature`'s floor is 0, so filling with 0 puts a fifth of the fronts at
the straightest value ever measured and displaces every measured front to
compensate. A clusterer can read that as a group of straight fronts rather than
short ones. The mean asserts nothing, at the cost of shrinking the column's
spread — the filled values have zero variance among themselves.

`curvature_direction` is the opposite case: its median is −0.0001 and half its
measured values are below 0, so a 0-fill lands at z = −0.01 and is
indistinguishable from mean-filling.

The indicator is what keeps either fill honest — without it, a front measured
at the fill value and a front nobody measured are the same point. Note it is
nearly redundant with `npix` and `length_km`, since every front with missing
curvature is short; `missing_indicator=False` is reasonable when those are
already selected.

The scale is computed **after** the drop, so a column's mean and sd describe
the rows that actually reach the clusterer.

### Curvature is the case this matters for

`mean_curvature` and `curvature_direction` are NaN for about 19% of fronts
(21,074 of 111,654 in the V5 SURF test store). That is not "this front is
straight" — straight is measurable and comes out near 0, and 402 fronts have
exactly 0.

NaN means the front was too short to measure. `calculate_front_curvature`
estimates curvature over a ±`window_size` stencil, `window_size=5`, so a
skeleton of 10 pixels or fewer yields no estimate at all. Every NaN front has
`npix` between 8 and 17 and length under 30 km.

So the missingness is nearly redundant with `npix` and `length_km`, which you
probably already selected. Dropping the **columns** costs you little; dropping
the **rows** costs a fifth of the dataset. Do not fill with 0 — that asserts
straightness about fronts nobody measured.

---

## What you get back

| attribute | |
|---|---|
| `X` | `(n_fronts, n_features)` float32, scaled, no NaN — this is what NEMI takes |
| `ids` | `date`, `label`, `name`, `time` for the same rows in the same order |
| `feature_names` | column names of `X` |
| `raw` | the same matrix before scaling, in physical units |
| `centre`, `scale` | what was subtracted and divided, per column |
| `dropped` | `{"rows": n, "columns": [...]}` |

`ds.to_frame()` puts the scaled features beside their ids for inspection.

Row *i* of `X` is the front at row *i* of `ids`; they are masked together, so a
cluster label can be attached straight back to `ids` and joined to the store on
`(date, label)`.
