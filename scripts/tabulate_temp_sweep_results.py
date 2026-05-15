#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create scenario-by-temperature performance tables from temp sweep benchmark output.

Rows: temperature values
Columns: scenario names
Cells: average metric value across repetitions
Outputs CSV tables only.
"""

import json
import os
from typing import Dict, List, Any

import pandas as pd

INPUT_FILE = "data/benchmark_temp_sweep_results.json"
OUTPUT_DIR = "data"

CATEGORY_MAP = {
    "focus_cost": "Cost",
    "focus_energy": "Energy",
    "focus_latency": "QoE",
    "balanced": "Balanced",
}
CATEGORY_ORDER = ["focus_cost", "focus_energy", "focus_latency", "balanced"]
METRICS = ["cost", "energy", "latency"]


def build_rows(data: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for temp_key, scenario_runs in data.items():
        try:
            temp_val = float(temp_key)
        except ValueError:
            continue

        for scenario, runs in scenario_runs.items():
            for run in runs:
                metrics = run.get("metrics", {})
                if not metrics:
                    continue

                row = {
                    "temperature": temp_val,
                    "scenario": scenario,
                    "scenario_label": CATEGORY_MAP.get(scenario, scenario),
                }
                for metric in METRICS:
                    row[metric] = metrics.get(metric)
                rows.append(row)

    return rows


def create_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    for metric in METRICS:
        pivot = pd.pivot_table(
            df,
            index="temperature",
            columns="scenario_label",
            values=metric,
            aggfunc="mean",
        )

        ordered_cols = [CATEGORY_MAP[c] for c in CATEGORY_ORDER if CATEGORY_MAP[c] in pivot.columns]
        pivot = pivot.reindex(columns=ordered_cols)
        pivot = pivot.sort_index()
        tables[metric] = pivot

    return tables


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Missing input file: {INPUT_FILE}")
        print("Run scripts/benchmark_temp_sweep.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = build_rows(data)
    if not rows:
        print("❌ No valid metric rows found in temp sweep results.")
        return

    df = pd.DataFrame(rows)
    tables = create_tables(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for metric, table in tables.items():
        csv_path = os.path.join(OUTPUT_DIR, f"temp_sweep_table_{metric}.csv")
        table.to_csv(csv_path, float_format="%.4f")
        print(f"✅ Saved {metric} table CSV: {csv_path}")

    print("\n--- Preview: Cost Table ---")
    print(tables["cost"].round(4))


if __name__ == "__main__":
    main()
