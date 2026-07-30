import numpy as np


def cluster_balance(labels):
    """How concentrated a labelling is: ``(top_fraction, effective_k)``.

    top_fraction : share of points in the largest cluster, so near 1 means a
        single cluster holds nearly everything.
    effective_k  : inverse Simpson index ``1 / sum(p**2)`` -- the number of
        clusters actually carrying the data.  Equals the cluster count when all
        clusters are the same size and falls toward 1 as one comes to dominate.

    Both are NaN for an empty labelling.
    """
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    if counts.size == 0:
        return float("nan"), float("nan")
    p = counts / counts.sum()
    return float(p.max()), float(1.0 / np.square(p).sum())
