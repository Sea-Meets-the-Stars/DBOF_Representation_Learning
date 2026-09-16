"""Turn a front store into the matrix NEMI clusters.

The cutout path has to build vectors out of images; a front is already a row of
numbers, so this is selection and scaling rather than embedding.  One row per
front, one column per requested feature, in the order the store returns them.

    source  = FrontDataSource("s3://.../fronts.zarr")
    raw     = RawFronts.load(source)
    ds      = raw.select(["geometry", "gradb2", "turner_angle"])
    nemi.run(ds.X, n=...)
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from front_finding.store import FrontStore

from field_scaling import (DIV_ABS, DIV_SIGNED, NORMALISED_EQUIVALENT, OMEGA)

#: Columns that identify a front rather than describe it.  They are carried on
#: the dataset as ids so a cluster can be traced back, never as features.
ID_COLUMNS = ("date", "label", "name", "time")

#: Where the front sits on the model grid.  Excluded by default: the bounding
#: box is an artefact of the raster, and centroid_lat/lon says the same thing
#: in physical units.
BBOX_COLUMNS = ("y0", "y1", "x0", "x1")

#: fronts() suffixes the properties table's npix, which duplicates geometry's
#: exactly.  Excluded so the same count is never fed in twice.
DUPLICATE_COLUMNS = ("flabel", "npix_prop")

#: Channels whose values span decades, so a log is applied before scaling.
LOG_PREFIX = "grad"

#: A property column is "{channel}_{stat}".  Recognising the statistic is what
#: lets a channel expand to a subset of them; a column that matches nothing
#: here -- every geometry column -- is not a statistic of anything and is never
#: filtered out.
STAT_SUFFIX = re.compile(r"_(mean|std|median|min|max|count|skew|p\d+)$")

#: Only the first two moments are divided by f.  A standard deviation takes
#: |f| -- a negative divisor would make it negative.  Skew is scale-invariant,
#: so dividing changes nothing.  min/max and the percentiles are order
#: statistics: dividing by a signed f swaps them end for end, and p90 would
#: need a p10 the store does not have.
SCALED_STATS = ("mean", "std")

#: Feature groups selectable by name.
GEOMETRY = "geometry"
PROPERTIES = "properties"
CROSS = "cross"


def _split_stat(column):
    """``('gradb2', 'mean')`` for a property column, ``(column, None)`` else."""
    match = STAT_SUFFIX.search(column)
    return (column[:match.start()], match.group(1)) if match else (column, None)


def _keep_stats(columns, stats):
    """*columns* narrowed to those statistics, keeping non-statistics."""
    if stats is None:
        return list(columns)
    stats = set(stats)
    return [c for c in columns
            if (stat := _split_stat(c)[1]) is None or stat in stats]


def _coriolis(table, equator_deg):
    """Signed f per front, floored at its *equator_deg* value.

    Taken from the front's own coriolis_f_mean, which co-location sampled over
    the same band as everything else.  There is no fallback: deriving f from
    the centroid would silently describe a different band, and a store without
    the channel should say so rather than approximate.

    The floor keeps the tropics finite -- f goes to zero at the equator and the
    kinematic fields would divide by ~0.
    """
    if "coriolis_f_mean" not in table.columns:
        raise KeyError(
            "div_by_f needs 'coriolis_f_mean', which this store's properties "
            "table does not have.  Add coriolis_f to the colocate channels and "
            "re-run that step."
        )
    f = table["coriolis_f_mean"].to_numpy(dtype="float64")
    floor = 2 * OMEGA * np.sin(np.deg2rad(equator_deg))
    return np.where(f < 0, -1.0, 1.0) * np.maximum(np.abs(f), floor)


def _safe_log10(a):
    """log10 with non-positive values sent to NaN rather than -inf.

    -inf would survive standardisation and make the whole column useless; NaN
    is visible to the nan_policy instead.
    """
    out = np.full(a.shape, np.nan, dtype="float64")
    np.log10(a, out=out, where=a > 0)
    return out


class FrontDataSource:
    """A front store, and what it offers as features."""

    def __init__(self, url, storage_options=None):
        self.url = url
        self.store = FrontStore.open(url, storage_options=storage_options)

    @property
    def dates(self):
        return self.store.dates

    def feature_groups(self, date=None):
        """``{group: [column, ...]}`` for the columns usable as features."""
        date = date or self.store.dates[0]
        geom = [c for c in self.store.geometry(date).columns
                if c not in ID_COLUMNS and c not in BBOX_COLUMNS]
        props = [c for c in self.store.properties(date).columns
                 if c not in DUPLICATE_COLUMNS]
        groups = {GEOMETRY: geom, PROPERTIES: props}
        if self.store.has_cross_properties(date):
            groups[CROSS] = [f"cross_{c}" for c in
                             self.store.cross_properties(date).columns
                             if c not in DUPLICATE_COLUMNS]
        return groups

    def print_available_features(self, date=None):
        for group, cols in self.feature_groups(date).items():
            print(f"{group} ({len(cols)})")
            print("   ", ", ".join(cols))


class RawFronts:
    """Every front of one or more snapshots, as the store returns them."""

    def __init__(self, table, groups, source_info=None):
        self.table = table
        self.groups = groups          # group name -> columns, from the store
        self.source_info = source_info or {}

    @classmethod
    def load(cls, source, dates=None, cross=True):
        """Read the store into one table.

        *cross* joins the cross-front statistics under a ``cross_`` prefix; it
        is silently skipped for snapshots that have none.
        """
        store = source.store
        dates = list(dates) if dates is not None else store.dates
        frames = []
        for date in dates:
            frame = store.fronts(date, cross=cross)
            frame.insert(0, "date", date)
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        # Group membership comes from the store's own tables rather than being
        # guessed from column names, which collide (cross_gradb2_mean vs
        # gradb2_mean) and would silently mis-file a new geometry column.
        groups = {g: [c for c in cols if c in table.columns]
                  for g, cols in source.feature_groups(dates[0]).items()}
        return cls(table, groups, {"url": source.url, "dates": dates,
                                   "n_fronts": len(table)})

    def __len__(self):
        return len(self.table)

    def resolve(self, features, stats=None):
        """Expand *features* to concrete column names, in the order given.

        An entry may be a group ('geometry', 'properties', 'cross'), a channel
        ('gradb2', which takes every gradb2_* statistic), or one column
        ('gradb2_mean').  Duplicates collapse, keeping first position.

        *stats* narrows what a group or channel expands to -- ('mean', 'std',
        'skew') to skip the order statistics.  It does not touch a column named
        outright, since naming it is already the choice, nor a geometry column,
        which is not a statistic of anything.
        """
        out = []
        for item in features:
            if item in self.groups:
                out.extend(_keep_stats(self.groups[item], stats))
            elif item in self.table.columns:
                out.append(item)
            else:
                matched = [c for c in self.table.columns
                           if c.startswith(f"{item}_") and c not in ID_COLUMNS]
                if not matched:
                    raise KeyError(
                        f"{item!r} is not a group, channel or column.  Groups: "
                        f"{sorted(self.groups)}.  Try "
                        f"FrontDataSource.print_available_features()."
                    )
                kept = _keep_stats(matched, stats)
                if not kept:
                    raise KeyError(
                        f"{item!r} has no {sorted(stats)} statistic; it offers "
                        f"{sorted(_split_stat(c)[1] for c in matched)}."
                    )
                out.extend(kept)
        return list(dict.fromkeys(out))

    def select(self, features, stats=None, scaling="standardize",
               log_channels=True, div_by_f=False, equator_deg=5.0,
               nan_policy="error", fill_value="mean", missing_indicator=True):
        """Build the dataset NEMI consumes.  See :class:`FrontDataset`."""
        columns = self.resolve(features, stats)
        return FrontDataset.build(self.table, columns, scaling=scaling,
                                  log_channels=log_channels,
                                  div_by_f=div_by_f, equator_deg=equator_deg,
                                  nan_policy=nan_policy,
                                  fill_value=fill_value,
                                  missing_indicator=missing_indicator,
                                  source_info=self.source_info)


class FrontDataset:
    """A front-by-feature matrix, scaled and ready to cluster.

    ``X`` is what goes to NEMI.  ``ids`` is the identifying columns for the
    same rows in the same order, so a cluster label can be joined back to the
    store; ``raw`` keeps the unscaled values for plotting in physical units.
    """

    def __init__(self, X, ids, feature_names, raw, centre, scale,
                 scaling, dropped, source_info=None):
        self.X = X
        self.ids = ids
        self.feature_names = feature_names
        self.raw = raw
        self.centre, self.scale = centre, scale
        self.scaling = scaling
        self.dropped = dropped
        self.source_info = source_info or {}

    @classmethod
    def build(cls, table, columns, scaling="standardize", log_channels=True,
              div_by_f=False, equator_deg=5.0,
              nan_policy="error", fill_value="mean", missing_indicator=True,
              source_info=None):
        """Select, log, drop NaN, then scale -- in that order.

        The scale is computed after the NaN drop so a column's statistics
        describe the rows that actually reach the clusterer.

        nan_policy defaults to 'error': dropping rows costs whole fronts and
        dropping columns costs a requested feature, and neither announces
        itself except in summary().  Which one is right depends on the
        selection, so it is asked for rather than guessed.

        'fill' keeps both, substituting *fill_value*: 'mean' for the column's
        own mean over the values it does have, or any number.  With
        *missing_indicator* it also adds a ``{column}_missing`` column, so a
        filled value stays distinguishable from a measured one.

        The choice is not cosmetic where the fill value carries meaning.
        mean_curvature's floor is 0, so filling it with 0 puts a fifth of the
        fronts at the straightest value ever measured, and a clusterer can
        read that as a group of straight fronts rather than short ones.  The
        mean asserts nothing, at the cost of shrinking the column's spread.
        """
        if nan_policy not in ("drop_rows", "drop_columns", "fill", "error"):
            raise ValueError("nan_policy must be 'error', 'fill', 'drop_rows' "
                             "or 'drop_columns'")
        if fill_value != "mean" and not isinstance(fill_value, (int, float)):
            raise ValueError("fill_value must be 'mean' or a number, got "
                             f"{fill_value!r}")
        if scaling not in ("standardize", "normalize", None):
            raise ValueError("scaling must be 'standardize', 'normalize' or None")

        values = table[columns].astype("float64").copy()
        if log_channels:
            for col in columns:
                if col.removeprefix("cross_").startswith(LOG_PREFIX):
                    values[col] = _safe_log10(values[col].to_numpy())

        scaled_by_f, unscaled_signed = [], []
        if div_by_f:
            f = np.abs(_coriolis(table, equator_deg))
            # columns is the resolved selection.  Split each into channel and
            # statistic -- 'cross_divergence_std' -> ('divergence', 'std').  A
            # geometry column splits to something that matches neither tuple
            # ('length_km' -> 'length', 'km'), so it falls through untouched.
            for col in columns:
                root, _, stat = col.removeprefix("cross_").rpartition("_")
                if root in DIV_SIGNED:
                    unscaled_signed.append(col)
                elif root in DIV_ABS and stat in SCALED_STATS:
                    values[col] = values[col] / f
                    scaled_by_f.append(col)
            if unscaled_signed:
                roots = sorted({c.removeprefix("cross_").rpartition("_")[0]
                                for c in unscaled_signed})
                swap = ", ".join(f"{r} -> {NORMALISED_EQUIVALENT[r]}"
                                 for r in roots if r in NORMALISED_EQUIVALENT)
                warnings.warn(
                    f"div_by_f left {len(unscaled_signed)} column(s) in raw "
                    f"units: {unscaled_signed}.  These need a SIGNED f, which "
                    f"cannot be applied to a stored statistic -- "
                    f"mean(x)/mean(f) is not mean(x/f), and the two part "
                    f"company where f varies across a front's band.  They "
                    f"still carry f's latitude dependence, so latitude can "
                    f"leak into the embedding.  The store already has the "
                    f"per-pixel quotient: {swap}.",
                    UserWarning, stacklevel=3)

        dropped = {"rows": 0, "columns": [], "filled": {},
                   "scaled_by_f": scaled_by_f,
                   "unscaled_signed": unscaled_signed}
        nan_by_column = values.isna().sum()
        if nan_policy == "error" and nan_by_column.any():
            offenders = nan_by_column[nan_by_column > 0]
            raise ValueError(
                f"{int(offenders.sum())} NaN across {len(offenders)} column(s): "
                f"{dict(offenders)}.  Pass nan_policy='drop_columns' to lose "
                f"those {len(offenders)} feature(s) and keep every front, or "
                f"'drop_rows' to lose "
                f"{int((~values.notna().all(axis=1)).sum())} of {len(values)} "
                f"fronts and keep the features, or 'fill' to keep both "
                f"(fill_value='mean' or a number)."
            )
        if nan_policy == "fill":
            indicators = {}
            for col in nan_by_column[nan_by_column > 0].index:
                missing = values[col].isna()
                # The mean is taken over the values the column does have, so a
                # filled row lands on the column's centre once scaled.
                value = (values[col].mean() if fill_value == "mean"
                         else float(fill_value))
                dropped["filled"][col] = {"n": int(missing.sum()),
                                          "value": float(value)}
                if missing_indicator:
                    indicators[f"{col}_missing"] = missing.astype("float64")
                values[col] = values[col].fillna(value)
            if indicators:
                # One concat rather than a column at a time: inserting eighty
                # indicators singly fragments the frame and pandas warns.
                values = pd.concat([values, pd.DataFrame(indicators,
                                                         index=values.index)],
                                   axis=1)
                columns = [*columns, *indicators]
        if nan_policy == "drop_columns":
            dropped["columns"] = list(nan_by_column[nan_by_column > 0].index)
            values = values.drop(columns=dropped["columns"])
            columns = [c for c in columns if c not in dropped["columns"]]
        keep = values.notna().all(axis=1)
        dropped["rows"] = int((~keep).sum())
        values = values[keep]

        if not len(values):
            raise ValueError(
                "every row has a NaN in the selected features; "
                f"NaN by column: {dict(nan_by_column[nan_by_column > 0])}"
            )

        raw = values.to_numpy()
        if scaling == "standardize":
            centre = raw.mean(axis=0)
            spread = raw.std(axis=0)
        elif scaling == "normalize":
            centre = raw.min(axis=0)
            spread = raw.max(axis=0) - centre
        else:
            centre = np.zeros(raw.shape[1])
            spread = np.ones(raw.shape[1])
        # A constant column has no spread to divide by and carries no signal;
        # leaving it at its centred value keeps it finite rather than NaN.
        spread = np.where(spread > 0, spread, 1.0)
        X = ((raw - centre) / spread).astype("float32")

        ids = table.loc[keep, [c for c in ID_COLUMNS if c in table.columns]]
        return cls(X, ids.reset_index(drop=True), list(columns), raw,
                   centre, spread, scaling, dropped, source_info)

    @classmethod
    def from_source(cls, source, features, dates=None, cross=True, **kwargs):
        """Load and select in one step."""
        return RawFronts.load(source, dates=dates, cross=cross).select(
            features, **kwargs)

    def __len__(self):
        return len(self.X)

    def to_frame(self):
        """Scaled features beside their ids, for inspection."""
        return pd.concat(
            [self.ids, pd.DataFrame(self.X, columns=self.feature_names)],
            axis=1)

    def summary(self):
        """One line.  Filled columns are counted, not listed -- a wide store
        fills dozens, and the dict is unreadable; see ``worst_filled()``."""
        out = [f"{len(self):,} fronts x {len(self.feature_names)} features",
               f"scaling={self.scaling}"]
        if self.dropped["rows"]:
            out.append(f"dropped {self.dropped['rows']:,} rows")
        if self.dropped["columns"]:
            out.append(f"dropped {len(self.dropped['columns'])} columns")
        if self.dropped["filled"]:
            worst, info = self.worst_filled()[0]
            out.append(f"filled {len(self.dropped['filled'])} columns"
                       f" (worst {worst} {100 * info['n'] / len(self):.0f}%)")
        if self.dropped["scaled_by_f"]:
            out.append(f"divided {len(self.dropped['scaled_by_f'])} by |f|")
        if self.dropped["unscaled_signed"]:
            out.append(f"{len(self.dropped['unscaled_signed'])} left in raw "
                       f"units (signed f)")
        return "  ".join(out)

    def worst_filled(self):
        """``(column, {"n", "value"})`` pairs, most-filled first.

        A column filled for most of the fronts is describing the fill rather
        than the data; this is how to notice before clustering on it.
        """
        return sorted(self.dropped["filled"].items(),
                      key=lambda kv: -kv[1]["n"])
