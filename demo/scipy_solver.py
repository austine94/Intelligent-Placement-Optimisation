"""Gurobi-free solver for the demo, using SciPy's HiGHS MILP backend.

Builds the same facility-location model as flexmedia_delivery_instance.exact_solve,
optionally over weighted viewer buckets, and writes the results back onto the
instance using the same attributes.
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from aggregation import cluster_viewers


def effective_costs(inst, alpha, beta):
    return beta * inst.consumer_costs + (1 - beta) * inst.energy_costs + alpha * inst.consumer_latencies


def _solve_milp(D, B, cost, w):
    """min D.y + sum_ik cost[i,k] x[i,k]; each bucket k to one node; sum_k w[k] x[i,k] <= B[i] y[i]."""
    m, K = cost.shape
    nvar = m + m * K
    obj = np.concatenate([D, cost.ravel()])
    A = lil_matrix((K + m * K + m, nvar))
    lb, ub = [], []
    row = 0
    for k in range(K):  # each bucket served by exactly one node
        for i in range(m):
            A[row, m + i * K + k] = 1
        lb.append(1); ub.append(1)
        row += 1
    for i in range(m):  # x[i,k] <= y[i]
        for k in range(K):
            A[row, m + i * K + k] = 1
            A[row, i] = -1
            lb.append(-np.inf); ub.append(0)
            row += 1
    for i in range(m):  # capacity in viewers
        for k in range(K):
            A[row, m + i * K + k] = w[k]
        A[row, i] = -B[i]
        lb.append(-np.inf); ub.append(0)
        row += 1
    res = milp(obj, constraints=LinearConstraint(A.tocsr(), lb, ub),
               integrality=np.ones(nvar), bounds=Bounds(0, 1),
               options={"mip_rel_gap": 1e-3, "time_limit": 30})
    if res.x is None:
        return res, None, None
    return res, res.x[:m] > 0.5, res.x[m:].reshape(m, K) > 0.5


def exact_solve_scipy(inst, alpha, beta, n_buckets=None, mode="cost-aware", labels=None):
    D = np.asarray(inst.datacentre_costs, dtype=float)
    B = np.asarray(inst.datacentre_capacities, dtype=float)
    eff = effective_costs(inst, alpha, beta)
    m, n = eff.shape

    if labels is None:
        labels = (cluster_viewers(inst.consumer_locations, eff, n_buckets, mode)
                  if n_buckets and n_buckets < n else np.arange(n))
    K = int(labels.max()) + 1
    onehot = np.zeros((n, K))
    onehot[np.arange(n), labels] = 1
    w = onehot.sum(axis=0)
    bucket_cost = eff @ onehot  # (m, K) sum of member costs, so the objective is exact

    t0 = time.perf_counter()
    res, y, x = _solve_milp(D, B, bucket_cost, w)
    inst.solve_seconds = time.perf_counter() - t0
    inst.n_buckets = K
    inst.bucket_labels = labels

    inst.optimisation_model_type = "Exact"
    inst.solve_status = res.status
    inst.solve_status_message = None
    if y is not None:
        if res.status != 0:
            inst.solve_status_message = f"HiGHS stopped early (status {res.status}); best feasible solution used."
        inst.exact_obj = float(res.fun)
        inst.open_centres = [i for i in range(m) if y[i]]
        node_of_bucket = x.argmax(axis=0)
        inst.assignments = {i: [int(j) for j in range(n) if node_of_bucket[labels[j]] == i]
                            for i in inst.open_centres}
    else:
        inst.solve_status_message = f"HiGHS ended with status {res.status}: {res.message}"
        inst.exact_obj = None
        inst.open_centres = None
        inst.assignments = None
