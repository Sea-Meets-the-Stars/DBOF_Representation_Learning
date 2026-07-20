import matplotlib.pyplot as plt
import numpy as np

def _make_ax(fig, dims, subplot=(1, 1, 1)):
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    return fig.add_subplot(*subplot, projection="3d" if dims == 3 else None)


def _scatter_embedding(ax, X_d, labels=None, categorical=True, dims=2, alpha=0.5, s=0.1, cmap="tab10"):
    kw = dict(s=s, alpha=alpha)
    if labels is not None:
        if categorical :
            kw.update(c=labels, cmap=cmap)
        else :
            kw.update(cmap=cmap)
    scatter = ax.scatter(*[X_d[:, i] for i in range(dims)], **kw)
    ax.set_xlabel("C1")
    ax.set_ylabel("C2")
    if dims == 3:
        ax.set_zlabel("C3")
    return scatter

def _set_limits(ax, X_d, dims, pct=1.0):
    """Frame the central (100 - 2*pct)% per axis, ignoring outliers."""
    setters = [ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if dims == 3 else [])
    for i, setlim in enumerate(setters):
        lo, hi = np.percentile(X_d[:, i], [pct, 100 - pct])
        setlim(lo, hi)

def _add_class_legend(ax, scatter, label_title):
    ax.add_artist(ax.legend(*scatter.legend_elements(), title=label_title))


def _labels_per_embedding(labels, n):
    """None, a shared (N,) array, or n per-embedding label arrays -> list of length n."""
    if labels is None:
        return [None] * n
    if len(labels) == n:            # (n, N) array or list of n label vectors
        return list(labels)
    return [labels] * n             # single shared (N,) array


def vis_dim_redux(X_d, labels=None, categorical=True, label_title="class", dims=2, alpha=0.5, cmap="tab10"):
    fig = plt.figure(figsize=(8, 6))
    ax = _make_ax(fig, dims)
    scatter = _scatter_embedding(ax, X_d, labels=labels, dims=dims, alpha=alpha, cmap=cmap, categorical=categorical)
    if labels is not None:
        _add_class_legend(ax, scatter, label_title)
    _set_limits(ax, X_d, dims, pct=1.0)
    plt.show()


def vis_dim_redux_list(embeddings, labels=None, categorical=True, titles=None, label_title="class",
                       dims=2, alpha=0.5, n_cols=3, cmap="tab10"):
    """Scatter a list/array of embeddings in a grid.

    embeddings : iterable of (N, dims) arrays, or an (M, N, dims) array.
    labels     : None, a shared (N,) array applied to every panel, or one label
                 array per embedding (e.g. member_clusters of shape (M, N)).
    titles     : optional per-panel titles.
    """
    n = len(embeddings)
    n_cols = min(n_cols, n)
    n_rows = -(-n // n_cols)                      # ceil
    per_labels = _labels_per_embedding(labels, n)

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    for i, X_d in enumerate(embeddings):
        ax = _make_ax(fig, dims, (n_rows, n_cols, i + 1))
        scatter = _scatter_embedding(ax, X_d, labels=per_labels[i], categorical=categorical, dims=dims, alpha=alpha, cmap=cmap)
        if titles is not None:
            ax.set_title(titles[i])
        if per_labels[i] is not None:
            _add_class_legend(ax, scatter, label_title)
        _set_limits(ax, X_d, dims, pct=1.0)
    fig.tight_layout()
    plt.show()