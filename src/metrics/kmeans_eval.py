import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def bic_aic(X, k, **gmm_kwargs):
    """Fit a k-component GMM to ``X`` and return its ``(bic, aic)``.

    X : (n_samples, n_features), any dimensionality.
    gmm_kwargs : forwarded to GaussianMixture, e.g. ``random_state``, or
        ``covariance_type='diag'`` for high-dimensional input, where the
        default full covariance costs O(k d^2) parameters.
    """
    gmm = GaussianMixture(n_components=k, init_params="kmeans", **gmm_kwargs).fit(X)
    return gmm.bic(X), gmm.aic(X)


def sse(X, labels):
    """Sum of squared distances from each sample to its cluster centroid,
    equivalent to a fitted KMeans' ``inertia_``.

    X      : (n_samples, n_features), any dimensionality.
    labels : (n_samples,) cluster assignment per sample.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        total += ((pts - pts.mean(axis=0, dtype=np.float64)) ** 2).sum(dtype=np.float64)
    return float(total)


def silhouette(X, labels, sample_size=10_000, seed=0):
    """Mean silhouette coefficient over samples, in [-1, 1].

    The exact score is O(n^2) in time, so above ``sample_size`` samples it is
    computed on a random subsample of X and labels together; ``sample_size=None``
    uses every sample.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    n = len(X)
    if sample_size is not None and n > sample_size:
        idx = np.random.default_rng(seed).choice(n, size=sample_size, replace=False)
        X, labels = X[idx], labels[idx]
    print(f"silhouette on {len(X):,} / {n:,} samples")
    return float(silhouette_score(X, labels))
