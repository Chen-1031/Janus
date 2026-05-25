"""Pure utilities for the Janus memory controller.

Kept free of any base-method dependency so it can be reused by any
base updater adapter.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import torch
from kmeans_pytorch import kmeans as _kmeans_pytorch


_KMEANS_PARAMS = set(inspect.signature(_kmeans_pytorch).parameters.keys())


def to_vec(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


# def _resolve_kmeans_device():
#     pref = os.getenv("JANUS_KMEANS_DEVICE", "").strip().lower()
#     if pref in {"cuda", "gpu"} and torch.cuda.is_available():
#         return torch.device("cuda")
#     # Default: CPU keeps k-means stable and cheap for small K_prime / n.
#     return torch.device("cpu")


def _resolve_kmeans_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine(a, b, eps: float = 1e-8) -> float:
    a = to_vec(a)
    b = to_vec(b)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def momentum_update(prev_m, z_t, beta: float):
    z_t = to_vec(z_t)
    if prev_m is None:
        return z_t.copy()
    prev_m = to_vec(prev_m)
    if prev_m.shape != z_t.shape:
        return z_t.copy()
    return beta * prev_m + (1.0 - beta) * z_t


def should_trigger_mmt(z_t, prev_m, tau: float) -> (bool, float):
    """Return (triggered, cosine_value). If prev_m is None, do not trigger."""
    if prev_m is None:
        return False, 0.0
    cos_val = cosine(z_t, prev_m)
    return (cos_val < float(tau)), cos_val


def run_kmeans(X, k: int, seed: int = 0, init_centers=None, distance: str = "euclidean"):
    """K-means over embeddings using ``kmeans_pytorch.kmeans``.

    Warm-start is supported via ``init_centers`` (passed as
    ``cluster_centers`` to kmeans_pytorch), matching the pattern shown in
    ``Method.md``.

    Returns
    -------
    centers : np.ndarray, shape (k_eff, d)
    labels  : np.ndarray, shape (n,), dtype int
    """
    X_np = np.asarray(X, dtype=float)
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)
    n = X_np.shape[0]
    if n == 0:
        return np.zeros((0, X_np.shape[1])), np.zeros(0, dtype=int)
    k_eff = max(1, min(int(k), n))

    # kmeans_pytorch has a known shape bug when num_clusters == 1
    # (its pairwise-distance tensor degenerates to 1D, then argmin(dim=1)
    # raises IndexError). For k==1 we can short-circuit: the only centroid
    # is the data mean and every point belongs to cluster 0.
    if k_eff == 1:
        centers = X_np.mean(axis=0, keepdims=True).astype(float)
        labels = np.zeros(n, dtype=int)
        return centers, labels

    device = _resolve_kmeans_device()
    X_t = torch.as_tensor(X_np, dtype=torch.float32, device=device)

    # Match the minimal call shown in Method.md, then opportunistically add
    # extra kwargs only when the installed ``kmeans_pytorch`` actually
    # supports them (different forks / versions expose different params).
    requested = {
        "X": X_t,
        "num_clusters": k_eff,
        "distance": distance,
        "device": device,
        "tqdm_flag": False,
        "iter_limit": 50,
        "seed": int(seed),
    }
    if init_centers is not None:
        init_np = np.asarray(init_centers, dtype=float)
        if (
            init_np.ndim == 2
            and init_np.shape[0] == k_eff
            and init_np.shape[1] == X_np.shape[1]
        ):
            requested["cluster_centers"] = torch.as_tensor(
                init_np, dtype=torch.float32, device=device
            )

    kwargs = {key: val for key, val in requested.items() if key in _KMEANS_PARAMS}

    # Best-effort determinism for variants that don't accept ``seed``.
    if "seed" not in _KMEANS_PARAMS:
        try:
            torch.manual_seed(int(seed))
        except Exception:
            pass

    labels_t, centers_t = _kmeans_pytorch(**kwargs)
    labels = labels_t.detach().cpu().numpy().astype(int)
    centers = centers_t.detach().cpu().numpy().astype(float)
    return centers, labels


class JanusState:
    """Per-task Janus controller state."""

    def __init__(self):
        self.step_count: int = 0
        self.trigger_count: int = 0

        # task records: {entry, question, output, gt, score, feedback, embedding}
        self.pending_tasks: list = []
        self.seen_tasks: list = []

        # coverage entries: {centroid, rep_task, embedding}
        self.coverage_set: list = []

        # boundary entries: {task, embedding, tag}
        self.boundary_set: list = []

        self.momentum = None  # np.ndarray | None
        self.centroids = None  # np.ndarray | None
        self.warmup_initialized: bool = False
