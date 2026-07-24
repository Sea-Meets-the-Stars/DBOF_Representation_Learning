"""Projection-quality metrics from Espadoto et al.'s quantitative DR survey.

Mt   trustworthiness    [0,1], higher better -- projection neighbours that are
                                                genuine data neighbours (few false neighbours).
Mc   continuity         [0,1], higher better -- data neighbours preserved in the projection.
Msig normalized_stress  lower better (0 = perfect) -- pairwise-distance preservation.
MNH  neighborhood_hit   [0,1], higher better -- projection neighbours sharing a point's label.

K defaults to 7 (survey convention).  X = data (R^n), P = projection P(D).
"""
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.manifold import trustworthiness as _sk_trustworthiness
from sklearn.neighbors import NearestNeighbors

DEFAULT_K = 7


def trustworthiness(X, P, k=DEFAULT_K):
    """Mt: proportion of each point's k projection-neighbours that are also among
    its k neighbours in the data (penalizes false neighbours)."""
    return float(_sk_trustworthiness(X, P, n_neighbors=k))


def continuity(X, P, k=DEFAULT_K):
    """Mc: proportion of each point's k data-neighbours preserved as projection
    neighbours (penalizes missing neighbours).  Trustworthiness with X and P
    swapped."""
    return float(_sk_trustworthiness(P, X, n_neighbors=k))


def normalized_stress(X, P, scale=True):
    """Msig: pairwise euclidean distance preservation, Sum((dn - a*dq)^2) / Sum(dn^2),
    lower is better (0 = perfect).

    scale=True fits the optimal isotropic factor ``a`` to the projection distances
    before comparing -- necessary because embeddings like UMAP have an arbitrary
    absolute scale.  scale=False (a=1) is the literal survey definition.
    """
    dn = pdist(X)
    dq = pdist(P)
    denom = np.dot(dn, dn)
    if denom == 0:
        return float("nan")
    a = np.dot(dn, dq) / np.dot(dq, dq) if (scale and np.any(dq)) else 1.0
    resid = dn - a * dq
    return float(np.dot(resid, resid) / denom)