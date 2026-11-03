import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "optimisation_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from flexmedia_delivery_instance import flexmedia_delivery_instance
from database import connect, now_iso
from config import BUCKETS_TXT, DEFAULT_BUCKETS
from scipy_solver import exact_solve_scipy


def policy_to_parameters(priority: str):
    text = priority.lower()
    if "energy" in text:
        return 0.0, 0.0
    if "latency" in text or "qoe" in text or "quality" in text:
        return 1.0, 0.5
    if "balanced" in text or "trade" in text:
        return 0.1, 0.5
    return 0.0, 1.0


def load_instance_from_db(db_path):
    with connect(db_path) as conn:
        viewers = conn.execute("SELECT * FROM viewers ORDER BY viewer_id").fetchall()
        nodes = conn.execute("SELECT * FROM compute_nodes ORDER BY node_id").fetchall()
        costs = conn.execute("SELECT * FROM viewer_node_costs ORDER BY node_id, viewer_id").fetchall()
    inst = flexmedia_delivery_instance(len(nodes), len(viewers), 1)
    inst.datacentre_locations = np.array([[r["longitude"], r["latitude"]] for r in nodes])
    inst.consumer_locations = np.array([[r["longitude"], r["latitude"]] for r in viewers])
    inst.datacentre_capacities = np.array([r["capacity"] for r in nodes])
    inst.datacentre_costs = np.array([r["provision_financial_cost"] for r in nodes])
    n_nodes, n_viewers = len(nodes), len(viewers)
    inst.consumer_costs = np.zeros((n_nodes, n_viewers))
    inst.energy_costs = np.zeros((n_nodes, n_viewers))
    inst.consumer_latencies = np.zeros((n_nodes, n_viewers))
    for row in costs:
        i, j = row["node_id"], row["viewer_id"]
        inst.consumer_costs[i, j] = row["financial_cost"]
        inst.energy_costs[i, j] = row["energy_cost"]
        inst.consumer_latencies[i, j] = row["latency"]
    return inst


def get_aggregation():
    """(n_buckets, mode) from buckets.txt, written by the agent CLI: e.g. "30 cost-aware"."""
    try:
        count, mode = (BUCKETS_TXT.read_text().split() + ["cost-aware"])[:2]
        return max(1, int(count)), (mode if mode in ("spatial", "cost-aware") else "cost-aware")
    except (OSError, ValueError):
        return DEFAULT_BUCKETS, "cost-aware"


def solve_and_store(db_path, priority):
    alpha, beta = policy_to_parameters(priority)
    inst = load_instance_from_db(db_path)
    n_buckets, mode = get_aggregation()
    exact_solve_scipy(inst, alpha=alpha, beta=beta, n_buckets=n_buckets, mode=mode)  # Gurobi-free (HiGHS)
    status = inst.solve_status_message or "optimal"
    metrics = None if inst.exact_obj is None else inst.get_exact_solution_breakdown()
    open_nodes = [] if inst.open_centres is None else inst.open_centres
    assignments = {} if inst.assignments is None else inst.assignments
    with connect(db_path) as conn:
        ingested_at = conn.execute("SELECT last_viewer_ingest FROM ingestion_summary WHERE id=1").fetchone()[0]
        cur = conn.execute(
            """INSERT INTO optimal_solutions
            (solved_at, priority, alpha, beta, objective_value, status, financial_cost, energy_cost, avg_latency, open_nodes, assignments,
             n_viewers, n_buckets, solve_seconds, bucket_labels, ingested_at, n_nodes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), priority, alpha, beta, inst.exact_obj, status,
             None if metrics is None else metrics["total_financial_cost"],
             None if metrics is None else metrics["total_energy_score"],
             None if metrics is None else metrics["avg_latency"],
             json.dumps(open_nodes), json.dumps(assignments),
             len(inst.bucket_labels), inst.n_buckets, inst.solve_seconds, json.dumps([int(b) for b in inst.bucket_labels]), ingested_at, len(inst.datacentre_costs)),
        )
        conn.commit()
        return cur.lastrowid
