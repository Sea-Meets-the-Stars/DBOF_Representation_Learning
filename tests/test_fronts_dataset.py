"""The contract NEMI binds to: a float matrix, no NaN, rows traceable back."""
import numpy as np
import pandas as pd
import pytest

from fronts_dataloader.fronts_dataset import (
    BBOX_COLUMNS, DUPLICATE_COLUMNS, FrontDataset, FrontDataSource,
    ID_COLUMNS, RawFronts)

from synthetic_fronts import DATE, DATE2, build_store


@pytest.fixture
def source(tmp_path):
    build_store(tmp_path / "fronts.zarr", n=40)
    return FrontDataSource(str(tmp_path / "fronts.zarr"))


@pytest.fixture
def raw(source):
    return RawFronts.load(source)


# ---------------------------------------------------------------------------
#  What the store offers
# ---------------------------------------------------------------------------

def test_identifiers_are_never_features(source):
    offered = {c for cols in source.feature_groups().values() for c in cols}
    assert not offered & set(ID_COLUMNS)
    assert not offered & set(BBOX_COLUMNS)


def test_the_duplicated_pixel_count_is_not_offered_twice(source):
    """fronts() renames properties.npix to npix_prop; it equals geometry's."""
    offered = [c for cols in source.feature_groups().values() for c in cols]
    assert not set(offered) & set(DUPLICATE_COLUMNS)


def test_groups_cover_geometry_properties_and_cross(source):
    groups = source.feature_groups()
    assert {"geometry", "properties", "cross"} == set(groups)
    assert "length_km" in groups["geometry"]
    assert "gradb2_mean" in groups["properties"]
    assert "cross_gradb2_mean" in groups["cross"]


def test_cross_is_absent_when_the_store_has_none(tmp_path):
    build_store(tmp_path / "s.zarr", n=10, cross=False)
    assert "cross" not in FrontDataSource(str(tmp_path / "s.zarr")).feature_groups()


# ---------------------------------------------------------------------------
#  Selecting features
# ---------------------------------------------------------------------------

def test_a_channel_expands_to_its_statistics(raw):
    assert raw.resolve(["gradb2"]) == ["gradb2_mean", "gradb2_std"]


def test_the_matrix_is_rectangular_under_every_policy(raw):
    """drop_columns narrows every row alike, so UMAP still sees one width."""
    for policy in ("drop_rows", "drop_columns"):
        ds = raw.select(["length_km", "mean_curvature"], nan_policy=policy)
        assert ds.X.ndim == 2
        assert ds.X.shape[1] == len(ds.feature_names)
        assert len(ds.X) == len(ds.ids)


def test_nan_is_an_error_unless_a_policy_is_chosen(raw):
    """Neither drop announces itself, so the default refuses instead."""
    with pytest.raises(ValueError, match="drop_columns"):
        raw.select(["mean_curvature"])


def test_a_group_expands_to_its_columns(raw):
    assert "length_km" in raw.resolve(["geometry"])


def test_one_column_can_be_named_directly(raw):
    assert raw.resolve(["gradb2_mean"]) == ["gradb2_mean"]


def test_selection_order_is_kept_and_duplicates_collapse(raw):
    assert raw.resolve(["gradb2_std", "gradb2", "gradb2_std"]) == [
        "gradb2_std", "gradb2_mean"]


def test_a_channel_does_not_pull_in_its_cross_twin(raw):
    """cross_gradb2_mean starts with 'cross_', so 'gradb2' must not match it."""
    assert all(not c.startswith("cross_") for c in raw.resolve(["gradb2"]))


def test_stats_narrows_what_a_channel_expands_to(raw):
    assert raw.resolve(["divergence"], stats=("mean", "std")) == [
        "divergence_mean", "divergence_std"]


def test_stats_narrows_a_group_too(raw):
    narrow = raw.resolve(["properties"], stats=("mean",))
    assert all(c.endswith("_mean") or "_" not in c.rstrip("x")
               for c in narrow if c not in ("npix",))
    assert len(narrow) < len(raw.resolve(["properties"]))


def test_stats_leaves_geometry_alone(raw):
    """A geometry column is not a statistic of anything."""
    narrow = raw.resolve(["geometry"], stats=("mean",))
    assert "length_km" in narrow
    assert "orientation" in narrow


def test_stats_does_not_touch_a_column_named_outright(raw):
    """Naming the column is already the choice."""
    assert raw.resolve(["divergence_min"], stats=("mean",)) == ["divergence_min"]


def test_a_channel_without_the_wanted_stat_says_what_it_has(raw):
    with pytest.raises(KeyError, match="median"):
        raw.resolve(["divergence"], stats=("median",))


def test_an_unknown_feature_names_the_groups(raw):
    with pytest.raises(KeyError, match="geometry"):
        raw.resolve(["no_such_thing"])


# ---------------------------------------------------------------------------
#  The matrix
# ---------------------------------------------------------------------------

def test_the_matrix_is_what_nemi_takes(raw):
    ds = raw.select(["gradb2", "length_km"])
    assert ds.X.ndim == 2
    assert ds.X.shape == (len(ds.ids), 3)
    assert ds.X.dtype == np.float32
    assert np.isfinite(ds.X).all()


def test_rows_stay_traceable_to_the_store(raw):
    ds = raw.select(["gradb2_mean"])
    assert list(ds.ids.columns) == ["date", "label", "name", "time"]
    assert len(ds.ids) == len(ds.X)
    assert ds.ids["label"].is_unique


def test_each_row_stays_with_its_own_front(raw):
    """X and ids are masked together; a desync would silently mislabel every
    cluster.  Tag each front with its own label, drop rows, then check the
    surviving values still match the ids beside them."""
    raw.table["tag"] = raw.table["label"].astype(float)
    ds = raw.select(["tag", "mean_curvature"], scaling=None,
                    nan_policy="drop_rows")
    assert ds.dropped["rows"] > 0                   # the drop really happened
    tag = ds.raw[:, ds.feature_names.index("tag")]
    assert (tag == ds.ids["label"].to_numpy()).all()


def test_standardize_centers_each_column(raw):
    ds = raw.select(["gradb2_mean", "length_km"], scaling="standardize")
    assert ds.X.mean(axis=0) == pytest.approx(0, abs=1e-5)
    assert ds.X.std(axis=0) == pytest.approx(1, abs=1e-5)


def test_normalize_puts_each_column_in_the_unit_interval(raw):
    ds = raw.select(["gradb2_mean", "length_km"], scaling="normalize")
    assert ds.X.min(axis=0) == pytest.approx(0, abs=1e-6)
    assert ds.X.max(axis=0) == pytest.approx(1, abs=1e-6)


def test_no_scaling_leaves_physical_values(raw):
    ds = raw.select(["length_km"], scaling=None, log_channels=False)
    assert ds.X.squeeze() == pytest.approx(ds.raw.squeeze(), rel=1e-5)


def test_a_constant_column_survives_scaling(raw):
    """Zero spread would divide by zero and NaN the whole column."""
    raw.table["flat"] = 3.0
    ds = raw.select(["flat"], scaling="standardize")
    assert np.isfinite(ds.X).all()


# ---------------------------------------------------------------------------
#  Logs and NaN
# ---------------------------------------------------------------------------

def test_gradient_channels_are_logged(raw):
    logged = raw.select(["gradb2_mean"], scaling=None)
    plain = raw.select(["gradb2_mean"], scaling=None, log_channels=False)
    assert logged.raw.max() < plain.raw.max()      # decades collapsed


def test_cross_gradient_channels_are_logged_too(raw):
    logged = raw.select(["cross_gradb2_mean"], scaling=None)
    plain = raw.select(["cross_gradb2_mean"], scaling=None, log_channels=False)
    assert logged.raw.max() < plain.raw.max()


def test_non_gradient_channels_are_left_alone(raw):
    ds = raw.select(["turner_angle_mean"], scaling=None)
    assert (ds.raw < 0).any()                      # a log would have NaNed these


def test_nan_rows_are_dropped_and_counted(raw):
    """A third of the synthetic fronts have no curvature, as in a real store."""
    ds = raw.select(["mean_curvature"], nan_policy="drop_rows")
    assert ds.dropped["rows"] > 0
    assert len(ds.X) == len(raw) - ds.dropped["rows"]
    assert np.isfinite(ds.X).all()


def test_nan_columns_can_be_dropped_instead(raw):
    ds = raw.select(["length_km", "mean_curvature"], nan_policy="drop_columns")
    assert ds.feature_names == ["length_km"]
    assert ds.dropped["rows"] == 0
    assert len(ds.X) == len(raw)


def test_fill_keeps_every_front_and_every_feature(raw):
    ds = raw.select(["length_km", "mean_curvature"], nan_policy="fill")
    assert len(ds.X) == len(raw)
    assert "mean_curvature" in ds.feature_names
    assert np.isfinite(ds.X).all()
    assert ds.dropped["filled"]["mean_curvature"]["n"] > 0


def test_fill_marks_which_values_were_filled(raw):
    """0 is the floor of mean_curvature, so a filled front must stay
    distinguishable from one genuinely measured at 0."""
    ds = raw.select(["mean_curvature"], nan_policy="fill", scaling=None,
                    fill_value=0.0)
    flag = ds.raw[:, ds.feature_names.index("mean_curvature_missing")]
    value = ds.raw[:, ds.feature_names.index("mean_curvature")]
    assert set(np.unique(flag)) == {0.0, 1.0}
    assert (value[flag == 1] == 0).all()
    assert flag.sum() == ds.dropped["filled"]["mean_curvature"]["n"]


def test_mean_fill_lands_on_the_columns_center(raw):
    """The default: a filled front sits at z=0 rather than at an extreme."""
    ds = raw.select(["mean_curvature"], nan_policy="fill",
                    fill_value="mean", missing_indicator=False)
    observed = raw.table["mean_curvature"].dropna()
    assert ds.dropped["filled"]["mean_curvature"]["value"] == pytest.approx(
        observed.mean())
    filled = ds.X[raw.table["mean_curvature"].isna().to_numpy()]
    assert filled.mean() == pytest.approx(0, abs=0.05)


def test_a_bad_fill_value_is_rejected(raw):
    with pytest.raises(ValueError, match="fill_value"):
        raw.select(["mean_curvature"], nan_policy="fill", fill_value="median")


def test_the_indicator_can_be_turned_off(raw):
    ds = raw.select(["mean_curvature"], nan_policy="fill",
                    missing_indicator=False)
    assert ds.feature_names == ["mean_curvature"]


def test_fill_value_is_configurable(raw):
    ds = raw.select(["mean_curvature"], nan_policy="fill", scaling=None,
                    fill_value=-99.0, missing_indicator=False)
    assert (ds.raw == -99.0).any()


def test_fill_stays_rectangular(raw):
    ds = raw.select(["length_km", "mean_curvature"], nan_policy="fill")
    assert ds.X.shape == (len(raw), len(ds.feature_names))


def test_nan_can_be_made_an_error(raw):
    with pytest.raises(ValueError, match="mean_curvature"):
        raw.select(["mean_curvature"], nan_policy="error")


def test_scale_is_computed_after_the_nan_drop(raw):
    """Otherwise the column's statistics describe rows that never reach NEMI."""
    ds = raw.select(["mean_curvature"], nan_policy="drop_rows")
    assert ds.X.mean() == pytest.approx(0, abs=1e-5)


def test_a_bad_policy_is_rejected(raw):
    with pytest.raises(ValueError, match="nan_policy"):
        raw.select(["length_km"], nan_policy="sometimes")


def test_a_bad_scaling_is_rejected(raw):
    with pytest.raises(ValueError, match="scaling"):
        raw.select(["length_km"], scaling="whiten")


# ---------------------------------------------------------------------------
#  Dividing by the Coriolis parameter
# ---------------------------------------------------------------------------

def test_div_by_f_scales_only_the_first_two_moments(raw):
    """Skew is scale-invariant and the order statistics would swap under a
    signed f, so only mean and std are touched."""
    cols = ["divergence_mean", "divergence_std", "divergence_skew",
            "divergence_min"]
    plain = raw.select(cols, scaling=None)
    scaled = raw.select(cols, scaling=None, div_by_f=True)
    i = {c: scaled.feature_names.index(c) for c in cols}
    assert not np.allclose(scaled.raw[:, i["divergence_mean"]],
                           plain.raw[:, i["divergence_mean"]])
    assert not np.allclose(scaled.raw[:, i["divergence_std"]],
                           plain.raw[:, i["divergence_std"]])
    assert np.allclose(scaled.raw[:, i["divergence_skew"]],
                       plain.raw[:, i["divergence_skew"]])
    assert np.allclose(scaled.raw[:, i["divergence_min"]],
                       plain.raw[:, i["divergence_min"]])
    assert set(scaled.dropped["scaled_by_f"]) == {"divergence_mean",
                                                  "divergence_std"}


def test_div_by_f_divides_by_the_magnitude(raw):
    """A standard deviation divided by a negative f would come out negative."""
    scaled = raw.select(["divergence_std"], scaling=None, div_by_f=True)
    assert (scaled.raw >= 0).all()


def test_div_by_f_leaves_a_signed_field_raw_and_says_so(raw):
    """A signed field cannot be scaled from a stored statistic, but it must not
    block the selection either -- warn loudly and pass it through."""
    plain = raw.select(["relative_vorticity_mean"], scaling=None)
    with pytest.warns(UserWarning, match="rossby_number"):
        scaled = raw.select(["relative_vorticity_mean"], scaling=None,
                            div_by_f=True)
    assert np.allclose(scaled.raw, plain.raw)          # untouched
    assert scaled.dropped["unscaled_signed"] == ["relative_vorticity_mean"]
    assert scaled.dropped["scaled_by_f"] == []


def test_a_signed_field_does_not_stop_the_others_scaling(raw):
    cols = ["relative_vorticity_mean", "divergence_mean"]
    with pytest.warns(UserWarning):
        ds = raw.select(cols, scaling=None, div_by_f=True)
    assert ds.dropped["scaled_by_f"] == ["divergence_mean"]
    assert ds.dropped["unscaled_signed"] == ["relative_vorticity_mean"]


def test_the_signed_warning_names_each_column_once(raw):
    """Two statistics of one channel are one problem, not two warnings."""
    with pytest.warns(UserWarning) as caught:
        raw.select(["relative_vorticity_mean", "relative_vorticity_std"],
                   scaling=None, div_by_f=True)
    assert len(caught) == 1


def test_div_by_f_leaves_other_channels_alone(raw):
    plain = raw.select(["gradb2_mean"], scaling=None)
    scaled = raw.select(["gradb2_mean"], scaling=None, div_by_f=True)
    assert np.allclose(scaled.raw, plain.raw)
    assert scaled.dropped["scaled_by_f"] == []


def test_div_by_f_needs_the_coriolis_channel(raw):
    """Deriving f from the centroid would describe a different band, so a
    store without the channel is an error rather than an approximation."""
    raw.table = raw.table.drop(columns=["coriolis_f_mean"])
    with pytest.raises(KeyError, match="coriolis_f"):
        raw.select(["divergence_mean"], div_by_f=True)


def test_div_by_f_floors_f_near_the_equator(raw):
    """f goes to zero at the equator; without the floor this divides by ~0."""
    raw.table["coriolis_f_mean"] = 0.0
    scaled = raw.select(["divergence_mean"], scaling=None, div_by_f=True,
                        equator_deg=5.0)
    assert np.isfinite(scaled.raw).all()


# ---------------------------------------------------------------------------
#  Several snapshots
# ---------------------------------------------------------------------------

def test_snapshots_concatenate_and_stay_labelled(tmp_path):
    build_store(tmp_path / "s.zarr", n=12, dates=(DATE, DATE2))
    ds = FrontDataset.from_source(
        FrontDataSource(str(tmp_path / "s.zarr")), ["gradb2_mean"])
    assert len(ds.X) == 24
    assert set(ds.ids["date"]) == {DATE, DATE2}
