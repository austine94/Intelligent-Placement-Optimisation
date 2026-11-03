"""Compare solving the full problem against bucketed solves (time and optimality gap).

Run from the repo root:  python demo/benchmark_aggregation.py --viewers 1000 --nodes 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimisation_engine"))
from flexmedia_delivery_instance import flexmedia_delivery_instance
from scipy_solver import exact_solve_scipy

import data_generator


def make_instance(n_viewers, n_nodes, seed, cost_model="structured"):
    np.random.seed(seed)
    inst = flexmedia_delivery_instance(n_nodes, n_viewers, 3)
    inst.datacentre_capacities = inst.datacentre_capacities * data_generator.capacity_scale(n_viewers, n_nodes)
    if cost_model == "structured":
        data_generator.structure_costs(inst)
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewers", type=int, default=1000)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--buckets", type=int, nargs="+", default=[10, 30, 60, 120])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--cost-model", choices=["structured", "random"], default="structured")
    ap.add_argument("--skip-full", action="store_true")
    a = ap.parse_args()

    rows = {}
    for seed in range(a.seeds):
        inst = make_instance(a.viewers, a.nodes, seed, a.cost_model)
        full = None
        if not a.skip_full:
            exact_solve_scipy(inst, a.alpha, a.beta)
            full = (inst.exact_obj, inst.solve_seconds)
            rows.setdefault("full", []).append((full[1], 0.0))
        for k in a.buckets:
            for mode in ("spatial", "cost-aware"):
                exact_solve_scipy(inst, a.alpha, a.beta, n_buckets=k, mode=mode)
                gap = float("nan") if full is None or inst.exact_obj is None else 100 * (inst.exact_obj - full[0]) / full[0]
                rows.setdefault(f"{mode:10s} K={k:<4d}", []).append((inst.solve_seconds, gap))
    print(f"{a.viewers} viewers, {a.nodes} nodes, {a.cost_model} costs, alpha={a.alpha}, beta={a.beta}, mean over {a.seeds} seeds")
    print(f"{'method':20s} {'solve s':>9s} {'gap %':>8s}")
    for name, vals in rows.items():
        t, g = np.mean(vals, axis=0)
        print(f"{name:20s} {t:9.3f} {g:8.2f}")


if __name__ == "__main__":
    main()
