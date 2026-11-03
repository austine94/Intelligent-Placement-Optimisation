import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "optimisation_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from flexmedia_delivery_instance import flexmedia_delivery_instance
from config import COMPUTE_NODES_CSV, RAW_DIR, VIEWERS_CSV
from database import ingest_csvs
from geo import LandMask, get_region, set_workload_region
from random_costs import (random_consumer_latencies, random_datacentre_capacities,
                          random_datacentre_costs)


def capacity_scale(n_viewers, n_nodes):
    """Engine capacities (10 edge / 100 core) suit ~24 viewers on 6 nodes; scale them with demand
    so larger workloads stay feasible (total capacity ~ 2-3x demand, as at the default size)."""
    return max(1.0, (n_viewers / 24) * (6 / n_nodes))


def structure_costs(inst, noise=0.15):
    """Make per-viewer delivery and energy costs a node-level rate with small per-viewer noise.

    The engine draws every (node, viewer) cost independently, so no two viewers are alike
    and nothing can be aggregated without losing optimality. Real delivery costs are set
    mostly by the serving node/network; this keeps that structure (latency stays geometric).
    """
    for name, lo, hi in (("consumer_costs", 0.01, 1.0), ("energy_costs", 0.001, 0.1)):
        current = getattr(inst, name)
        m, n = current.shape
        base = np.random.uniform(lo, hi, size=(m, 1))
        noisy = base * (1 + noise * np.random.randn(m, n))
        setattr(inst, name, np.clip(noisy, lo, None))


def place_on_land(inst, land, n_nodes, n_viewers, n_hotspots, min_radius=0.01, min_var=0.01, max_var=0.1):
    """Re-place nodes and viewers on land, then rebuild the location-dependent
    quantities (core/edge costs and capacities, latencies) with the engine's own functions."""
    nodes = land.sample_uniform(1)
    while len(nodes) < n_nodes:
        cand = land.sample_uniform(1)
        if np.all(np.linalg.norm(nodes - cand, axis=1) > min_radius):
            nodes = np.vstack([nodes, cand])
    weights = np.random.uniform(0, 1, n_hotspots)
    weights /= weights.sum()
    means = land.sample_uniform(n_hotspots)
    var_vals = np.random.uniform(min_var, max_var, n_hotspots)
    which = np.random.choice(n_hotspots, n_viewers, True, weights)
    viewers = np.array([land.sample_near(means[h], var_vals[h]) for h in which])
    inst.datacentre_locations = nodes
    inst.consumer_locations = viewers
    inst.datacentre_costs = random_datacentre_costs(nodes, 10, 100, 1, 10)
    inst.datacentre_capacities = random_datacentre_capacities(nodes, 10, 100) * capacity_scale(n_viewers, n_nodes)
    inst.consumer_latencies = random_consumer_latencies(nodes, viewers, 1.0)


def generate_snapshot(n_viewers=24, n_nodes=6, n_hotspots=3, missing_rate=0.05, seed=None, cost_model="structured"):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    inst = flexmedia_delivery_instance(n_datacentres=n_nodes, n_consumers=n_viewers, n_hotspots=n_hotspots)
    inst.datacentre_capacities = inst.datacentre_capacities * capacity_scale(n_viewers, n_nodes)
    if cost_model == "structured":
        structure_costs(inst)
    region = get_region()
    if region != "unit":
        place_on_land(inst, LandMask(region), n_nodes, n_viewers, n_hotspots)
    set_workload_region(region)

    with open(VIEWERS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["viewer_id", "longitude", "latitude"])
        writer.writeheader()
        for viewer_id, (lon, lat) in enumerate(inst.consumer_locations):
            writer.writerow({"viewer_id": viewer_id, "longitude": lon, "latitude": lat})

    fields = ["node_id", "longitude", "latitude", "capacity", "provision_financial_cost", "provision_energy_cost"]
    for viewer_id in range(n_viewers):
        fields.extend([f"viewer_{viewer_id}_financial_cost", f"viewer_{viewer_id}_energy_cost", f"viewer_{viewer_id}_latency"])

    def maybe_missing(value):
        return "" if random.random() < missing_rate else value

    with open(COMPUTE_NODES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for node_id, (lon, lat) in enumerate(inst.datacentre_locations):
            row = {
                "node_id": node_id,
                "longitude": lon,
                "latitude": lat,
                "capacity": maybe_missing(int(max(1, round(inst.datacentre_capacities[node_id])))),
                "provision_financial_cost": maybe_missing(float(inst.datacentre_costs[node_id])),
                "provision_energy_cost": maybe_missing(float(np.mean(inst.energy_costs[node_id, :]))),
            }
            for viewer_id in range(n_viewers):
                row[f"viewer_{viewer_id}_financial_cost"] = maybe_missing(float(inst.consumer_costs[node_id, viewer_id]))
                row[f"viewer_{viewer_id}_energy_cost"] = maybe_missing(float(inst.energy_costs[node_id, viewer_id]))
                row[f"viewer_{viewer_id}_latency"] = maybe_missing(float(inst.consumer_latencies[node_id, viewer_id]))
            writer.writerow(row)
    return ingest_csvs(VIEWERS_CSV, COMPUTE_NODES_CSV)


def main():
    parser = argparse.ArgumentParser(description="Generate changing demo workloads and ingest them into SQLite.")
    parser.add_argument("--loop", action="store_true", help="Regenerate and ingest every interval seconds.")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--viewers", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=6)
    args = parser.parse_args()
    while True:
        summary = generate_snapshot(args.viewers, args.nodes)
        print(f"Generated and ingested workload: {summary}")
        if not args.loop:
            break
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
