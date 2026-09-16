"""Sweep NEMI's embedding and clustering parameters over one feature matrix.

Every combination of the two grids is fitted and scored, and the result carries
enough to plot without re-running anything: the scores as a table, and the
embeddings and cluster labels behind them.

    grid    = {"n_neighbors": [15, 50], "min_dist": [0.0, 0.1]}
    cluster = {"method": ["hdbscan"], "min_cluster_size": [200, 400],
               "min_samples": [None]}
    result  = run_sweep(ds.X, grid, cluster, device="gpu")

    result.table                      # one row per (embedding x clustering)
    result.best("trustworthiness")

An embedding is fitted once per embedding config and reused for every
clustering config on top of it, which is where most of the time is saved: UMAP
dominates, and re-fitting it per clustering config would multiply the run by
the size of the clustering grid.
"""
from __future__ import annotations

import gc
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metrics import cluster_eval, kmeans_eval
from metrics import projection_metrics as pm
from nemi import SingleNemi

#: Scores taken from the embedding alone, before any clustering.
PROJECTION_METRICS = ("trustworthiness", "continuity", "normalized_stress")

#: Scores that need cluster labels.
CLUSTER_METRICS = ("found_k", "noise_%", "top_cluster_%", "effective_k",
                   "silhouette")


def expand_grid(grid):
    """``{param: [values]}`` -> a dict per full-factorial combination.

    ``None`` or an empty grid yields a single empty config, so a sweep with no
    clustering grid still runs its embeddings once.
    """
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def _free_gpu():
    """Drop references and return pooled GPU blocks between fits.

    cupy is absent on a CPU-only install, where there is nothing to free.
    """
    gc.collect()
    try:
        import cupy as cp
    except ImportError:
        return
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _n_clusters(labels):
    """Cluster count, ignoring NaN and the -1 noise label."""
    finite = np.asarray(labels, dtype=float)
    finite = finite[np.isfinite(finite) & (finite >= 0)]
    return int(finite.max()) + 1 if finite.size else 0


def _fit_embedding(X, params, n_components, device):
    """One UMAP embedding of X."""
    single = SingleNemi(params={
        "device": device,
        "embedding_dict": {"n_components": n_components, **params}})
    single.fit_embedding(X)
    return np.asarray(single.embedding)


def _cluster(embedding, params, device):
    """Cluster an already-fitted embedding, without re-embedding it."""
    single = SingleNemi(params={"device": device,
                                "clustering_dict": dict(params)})
    single.embedding = np.asarray(embedding)
    return np.asarray(single.predict_clusters())


@dataclass
class SweepResult:
    """Scores for every configuration, and what produced them.

    ``table`` is one row per (embedding config x clustering config), carrying
    the parameters that made it beside the metrics.  ``embeddings`` and
    ``labels`` are parallel to its rows, so a plot can show the run behind a
    cell without fitting anything again.
    """

    table: pd.DataFrame
    embeddings: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    embedding_params: tuple = ()
    clustering_params: tuple = ()
    failures: list = field(default_factory=list)
    #: Embeddings actually fitted.  Fewer than len(table) whenever a clustering
    #: grid reuses each one, which is the point of fitting them once.
    n_fitted: int = 0

    def __len__(self):
        return len(self.table)

    @property
    def swept(self):
        """Parameters that actually varied -- the ones worth an axis."""
        return tuple(c for c in (*self.embedding_params, *self.clustering_params)
                     if self.table[c].nunique(dropna=False) > 1)

    def metrics(self):
        """Metric columns present, in the order they are computed."""
        return [m for m in (*PROJECTION_METRICS, *CLUSTER_METRICS)
                if m in self.table.columns]

    def best(self, metric, n=5, ascending=None):
        """The *n* best rows by *metric*.

        Lower is better for normalized_stress and noise_%; higher for the
        rest.  Pass *ascending* to override.
        """
        if metric not in self.table.columns:
            raise KeyError(f"{metric!r} not in results; have {self.metrics()}")
        if ascending is None:
            ascending = metric in ("normalized_stress", "noise_%")
        return self.table.sort_values(metric, ascending=ascending).head(n)

    def summary(self):
        out = [f"{len(self)} configurations",
               f"{self.n_fitted} embeddings fitted"]
        if self.swept:
            out.append("swept " + ", ".join(self.swept))
        if self.failures:
            out.append(f"{len(self.failures)} failed")
        return "  ".join(out)


def run_sweep(X, embedding_grid, clustering_grid=None, *, n_components=3,
              device="gpu", fit_size=None, metric_sample=3000,
              silhouette=False, k=pm.DEFAULT_K, keep_size=30000,
              seed=0, verbose=True):
    """Fit every embedding config, cluster each with every clustering config.

    Parameters
    ----------
    X : np.ndarray
        ``(n_samples, n_features)`` -- e.g. ``FrontDataset.X``.
    embedding_grid, clustering_grid : dict
        ``{param: [values]}``, expanded full-factorially.  The clustering grid
        must name its ``method``; omit it entirely to score embeddings alone.
    n_components : int
        Embedding dimensionality, held fixed across the sweep.
    fit_size : int, optional
        Subsample this many rows before fitting, so a grid stays affordable on
        a large dataset.  None fits all of X.
    metric_sample : int
        Rows used for the projection metrics, which are O(n^2) in neighbours.
    silhouette : bool
        Also score each clustering.  Needs labels, so it does nothing without a
        clustering grid.
    keep_size : int, optional
        Rows of each embedding kept for plotting.  None keeps all; the point is
        to avoid holding a full copy per configuration.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)

    fit_idx = (rng.choice(len(X), size=fit_size, replace=False)
               if fit_size and fit_size < len(X) else np.arange(len(X)))
    Xfit = X[fit_idx]

    # The projection metrics compare neighbourhoods in both spaces, so they run
    # on a sample rather than the whole fit.
    m_idx = rng.choice(len(Xfit), size=min(metric_sample, len(Xfit)),
                       replace=False)
    Xm = Xfit[m_idx]

    keep = (rng.choice(len(Xfit), size=min(keep_size, len(Xfit)), replace=False)
            if keep_size else np.arange(len(Xfit)))

    embed_configs = expand_grid(embedding_grid)
    cluster_configs = expand_grid(clustering_grid)
    rows, embeddings, all_labels, failures = [], [], [], []
    n_fitted = 0

    for embed_params in embed_configs:
        if verbose:
            print(f"embedding {embed_params}")
        try:
            E = _fit_embedding(Xfit, embed_params, n_components, device)
        except Exception as exc:               # a bad config, not a bug
            failures.append({**embed_params,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"  embedding failed: {type(exc).__name__}: {exc}")
            _free_gpu()
            continue
        n_fitted += 1

        Em = E[m_idx]
        proj = {"trustworthiness": pm.trustworthiness(Xm, Em, k),
                "continuity": pm.continuity(Xm, Em, k),
                "normalized_stress": pm.normalized_stress(Xm, Em)}

        for cluster_params in cluster_configs:
            row = {**embed_params, **cluster_params, **proj}
            kept_labels = None
            if cluster_params:
                try:
                    lab = _cluster(E, cluster_params, device)
                    row["found_k"] = _n_clusters(lab)
                    row["noise_%"] = 100 * float(np.mean(lab < 0))
                    top, eff_k = cluster_eval.cluster_balance(lab[lab >= 0])
                    row["top_cluster_%"] = 100 * top
                    row["effective_k"] = eff_k
                    if silhouette:
                        assigned = lab >= 0
                        row["silhouette"] = kmeans_eval.silhouette(
                            E[assigned], lab[assigned],
                            sample_size=metric_sample, seed=seed)
                    kept_labels = lab[keep]
                except Exception as exc:
                    failures.append({**embed_params, **cluster_params,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  clustering failed {cluster_params}: "
                          f"{type(exc).__name__}: {exc}")
                finally:
                    _free_gpu()

            rows.append(row)
            embeddings.append(E[keep])
            all_labels.append(kept_labels)
            if verbose:
                shown = {k_: v for k_, v in row.items() if k_ not in embed_params}
                print("   " + "  ".join(
                    f"{k_}={v:.3f}" if isinstance(v, float) else f"{k_}={v}"
                    for k_, v in shown.items()))
        _free_gpu()

    return SweepResult(pd.DataFrame(rows), embeddings, all_labels,
                       tuple(embedding_grid or ()),
                       tuple(clustering_grid or ()), failures, n_fitted)
