import json
from pathlib import Path

import math

from config import MAP_DATA, OUTPUT_DIR
from geo import REGIONS, get_workload_region
from database import connect

def _draw_map(plt, region):
    """Draw the basemap on the current axes; returns a function mapping unit coords to lon/lat."""
    (lon0, lon1), (lat0, lat1) = region["lon"], region["lat"]
    ax = plt.gca()
    lat_c = (lat0 + lat1) / 2
    ax.set_facecolor("#dbeaf5")
    data = json.loads(MAP_DATA.read_text())
    for ring in data["countries"]:
        xs, ys = zip(*ring)
        ax.fill(xs, ys, facecolor="#f4f1ea", edgecolor="#9aa5b1", linewidth=0.6, zorder=1)
    for ring in data["districts"]:
        xs, ys = zip(*ring)
        ax.plot(xs, ys, color="#c9ced6", linewidth=0.4, zorder=2)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / math.cos(math.radians(lat_c)))  # equal ground distance on both axes
    km = region["scale_km"]
    bar = km / (111.2 * math.cos(math.radians(lat_c)))
    x0 = lon0 + 0.04 * (lon1 - lon0); y0 = lat0 + 0.05 * (lat1 - lat0)
    ax.plot([x0, x0 + bar], [y0, y0], color="black", linewidth=2.5, zorder=6)
    ax.text(x0 + bar / 2, y0 + 0.015 * (lat1 - lat0), f"{km} km", ha="center", fontsize=8, zorder=6)
    return lambda x, y: (lon0 + x * (lon1 - lon0), lat0 + y * (lat1 - lat0))


def render_latest_solution(db_path, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        sol = conn.execute("SELECT * FROM optimal_solutions ORDER BY solution_id DESC LIMIT 1").fetchone()
        viewers = conn.execute("SELECT * FROM viewers ORDER BY viewer_id").fetchall()
        nodes = conn.execute("SELECT * FROM compute_nodes ORDER BY node_id").fetchall()
        ingested_at = conn.execute("SELECT last_viewer_ingest FROM ingestion_summary WHERE id=1").fetchone()[0]
    if sol is None:
        return "No solution available."
    if sol["ingested_at"] != ingested_at and (output_dir / "latest_solution.png").exists():
        # A new workload has been ingested but not solved yet; drawing the old solution on
        # the new viewers would be wrong, so keep the last consistent picture.
        return "Workload not solved yet; keeping the previous plot."
    assignments = {int(k): v for k, v in json.loads(sol["assignments"]).items()}
    open_nodes = set(json.loads(sol["open_nodes"]))
    lines = [
        "# Latest Agentic Compute Provisioning Solution",
        "",
        f"Solved at: {sol['solved_at']}",
        f"Priority: {sol['priority']}",
        f"Weights: alpha={sol['alpha']}, beta={sol['beta']}",
        f"Objective: {sol['objective_value']}",
        f"Financial cost: {sol['financial_cost']}",
        f"Energy score: {sol['energy_cost']}",
        f"Average latency: {sol['avg_latency']}",
        f"Open nodes: {sorted(open_nodes)}",
        f"Viewers: {sol['n_viewers']}, solved as {sol['n_buckets']} weighted buckets in {0 if sol['solve_seconds'] is None else round(sol['solve_seconds'], 3)} s",
        "",
        "## Allocation summary",
    ]
    for node in nodes:
        assigned = assignments.get(node["node_id"], [])
        status = "OPEN" if node["node_id"] in open_nodes else "closed"
        lines.append(f"- Node {node['node_id']} ({status}, capacity {node['capacity']}): {len(assigned)} viewers")
    path = output_dir / "latest_solution.md"
    path.write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-GUI backend: plotting runs off the main thread
        import matplotlib.pyplot as plt
        region_name = get_workload_region()
        region = REGIONS[region_name]
        if region_name == "unit":
            plt.figure(figsize=(7, 6))
            to_xy = lambda x, y: (x, y)
            plt.xlim(0, 1); plt.ylim(0, 1); plt.xlabel("longitude"); plt.ylabel("latitude")
        else:
            (lon0, lon1), (lat0, lat1) = region["lon"], region["lat"]
            height = 6 * (lat1 - lat0) / ((lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2)))
            plt.figure(figsize=(6, max(3, min(7, height))))
            to_xy = _draw_map(plt, region)
            plt.xticks([]); plt.yticks([])
        colour = lambda node_id: plt.cm.tab20(node_id % 20)
        owner = {v: n for n, members in assignments.items() for v in members}
        many = len(viewers) > 100
        for idx, viewer in enumerate(viewers):
            vx, vy = to_xy(viewer["longitude"], viewer["latitude"])
            plt.scatter(vx, vy, marker="o", s=8 if many else 25, c=[colour(owner[idx])] if idx in owner else "tab:red",
                        alpha=0.55 if many else 0.8, linewidths=0, zorder=4)
        for node in nodes:
            is_open = node["node_id"] in open_nodes
            nx, ny = to_xy(node["longitude"], node["latitude"])
            plt.scatter(nx, ny, marker="s" if is_open else "x", s=170 if is_open else 80, c=[colour(node["node_id"])],
                        edgecolors="black" if is_open else None, linewidths=1.0, zorder=6)
            plt.text(nx, ny, f"N{node['node_id']}", zorder=7)
        plt.title(f"Optimised allocation: {sol['priority']}")
        plt.tight_layout()
        png = output_dir / "latest_solution.png"
        plt.savefig(png, dpi=200)
        plt.close()
        return f"Wrote {path} and {png}"
    except Exception as exc:
        return f"Wrote {path}; plot skipped: {exc}"
