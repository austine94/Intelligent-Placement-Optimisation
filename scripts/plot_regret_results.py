#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot regret benchmark results by scenario.

Reads data/benchmark_regret_results.json and creates:
- regret_absolute.png
- regret_percent.png
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_FILE = "data/benchmark_regret_results.json"

CATEGORY_MAP = {
    "focus_cost": "Cost",
    "focus_energy": "Energy",
    "focus_latency": "QoE",
    "balanced": "Balanced",
}
ORDER = ["Cost", "Energy", "QoE", "Balanced"]


def load_regret_dataframe() -> pd.DataFrame | None:
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        print("Run scripts/benchmark_regret.py first.")
        return None

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    skipped = 0
    total = 0

    for category, runs in data.items():
        for run in runs:
            total += 1
            abs_regret = run.get("objective_regret_absolute")
            pct_regret = run.get("objective_regret_percent")

            if abs_regret is None or pct_regret is None:
                skipped += 1
                continue

            rows.append(
                {
                    "Scenario": CATEGORY_MAP.get(category, category),
                    "Objective Regret (Absolute)": abs_regret,
                    "Objective Regret (%)": pct_regret,
                }
            )

    if not rows:
        print("❌ No valid regret rows found.")
        return None

    print(f"Processed {len(rows)} valid regret rows.")
    if skipped > 0:
        print(f"Skipped {skipped}/{total} runs due to missing regret fields.")

    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, y_col: str, title: str, output_file: str, color: str) -> None:
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Scenario", y=y_col, order=ORDER, color=color, errorbar="sd", capsize=0.1)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Scenario", fontsize=12)
    plt.ylabel(y_col, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {output_file}")


def main() -> None:
    sns.set_theme(style="whitegrid")

    df = load_regret_dataframe()
    if df is None or df.empty:
        return

    # Text summary
    summary = df.groupby("Scenario")[["Objective Regret (Absolute)", "Objective Regret (%)"]].mean()
    print("\n--- Mean Regret by Scenario ---")
    print(summary.round(4))
    print("-" * 40)

    plot_metric(
        df=df,
        y_col="Objective Regret (Absolute)",
        title="Objective Regret (Absolute) by Scenario",
        output_file="regret_absolute.png",
        color="#9467bd",
    )

    plot_metric(
        df=df,
        y_col="Objective Regret (%)",
        title="Objective Regret (%) by Scenario",
        output_file="regret_percent.png",
        color="#ff7f0e",
    )


if __name__ == "__main__":
    main()
