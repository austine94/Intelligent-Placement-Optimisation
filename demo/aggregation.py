"""Aggregate many viewers into a small number of weighted buckets.

A bucket is served by a single compute node, so a bucket solution is a valid
assignment of every viewer in it, and its true cost is the sum of its members'
costs. Only optimality is traded away, never feasibility.
"""
import numpy as np
from scipy.cluster.vq import kmeans2


def _zscore(a):
    sd = a.std(axis=0)
    return (a - a.mean(axis=0)) / np.where(sd == 0, 1, sd)


def cluster_viewers(locations, eff_cost, n_buckets, mode="cost-aware", seed=0):
    """Return an integer bucket label per viewer (labels are 0..K-1, none empty).

    locations: (n, 2) viewer coordinates.
    eff_cost:  (m, n) cost of serving each viewer from each node under the active policy.
    mode:      "spatial" clusters on location only; "cost-aware" also clusters on each
               viewer's cost profile across nodes, so bucket members are interchangeable
               for the objective as well as close together.
    """
    n = len(locations)
    if n_buckets >= n:
        return np.arange(n)
    loc = _zscore(np.asarray(locations, dtype=float))
    if mode == "spatial":
        feats = loc
    else:
        prof = _zscore(np.asarray(eff_cost, dtype=float).T)
        # give location and cost profile equal total weight
        feats = np.hstack([loc / np.sqrt(loc.shape[1]), prof / np.sqrt(prof.shape[1])])
    _, labels = kmeans2(feats, n_buckets, minit="++", seed=np.random.default_rng(seed))
    _, labels = np.unique(labels, return_inverse=True)  # drop empty clusters
    return labels
