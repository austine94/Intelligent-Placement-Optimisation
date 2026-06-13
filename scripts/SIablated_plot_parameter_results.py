#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:04:39 2026

@author: localadmin
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
INPUT_FILE = "data/SIablated_parameter_choices.json"

# Clean labels for the plot
CATEGORY_MAP = {
    "focus_cost": "Cost",
    "focus_energy": "Energy",
    "focus_latency": "Latency",
    "tradeoff_cost_energy": "Cost-Energy",
    "tradeoff_cost_latency": "Cost-Latency",
    "tradeoff_energy_latency": "Energy-Latency",
}

# Enforce consistent order on X-axis
ORDER = ["Cost", "Energy", "Latency", "Cost-Energy", "Cost-Latency", "Energy-Latency"]

def clean_val(val):
    """Helper to safely convert strings/nulls to floats."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def plot_parameters():
    if not os.path.exists(INPUT_FILE):
        print("❌ File not found. Run benchmark_parameters.py first!")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    rows = []
    skipped = 0

    for category, runs in data.items():
        for run in runs:
            if 'error' in run: 
                skipped += 1
                continue 
            
            # Clean data
            alpha = clean_val(run.get('alpha'))
            beta = clean_val(run.get('beta'))

            if alpha is None or beta is None:
                skipped += 1
                continue

            rows.append({
                "Strategy": CATEGORY_MAP.get(category, category),
                "Alpha": alpha,
                "Beta": beta
            })

    if not rows:
        print("❌ No valid data found to plot.")
        return

    df = pd.DataFrame(rows)
    print(f"📉 Processed {len(rows)} valid points. (Skipped {skipped} errors)")
    
    # Setup Style
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: ALPHA (Latency Weight) ---
    plt.figure(figsize=(8, 6))
    
    # Boxplot (Distribution summary)
    sns.boxplot(
        data=df, x="Strategy", y="Alpha", order=ORDER,
        hue="Strategy", palette="viridis", dodge=False, showfliers=False,
        boxprops={'alpha': 0.3} 
    )
    # Stripplot (Individual dots)
    sns.stripplot(
        data=df, x="Strategy", y="Alpha", order=ORDER,
        hue="Strategy", palette="viridis", jitter=True, alpha=0.6, legend=False
    )
    
    plt.xlabel("Scenario", fontsize=16)
    plt.ylabel("Alpha Value", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    
    filename_alpha = "SIablated_parameter_consistency_alpha.png"
    plt.tight_layout()
    plt.savefig(filename_alpha, dpi=300)
    print(f"✅ Saved Alpha plot: {filename_alpha}")
    plt.close()

    # --- PLOT 2: BETA (Cost vs Energy) ---
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(
        data=df, x="Strategy", y="Beta", order=ORDER,
        hue="Strategy", palette="magma", dodge=False, showfliers=False,
        boxprops={'alpha': 0.3}
    )
    sns.stripplot(
        data=df, x="Strategy", y="Beta", order=ORDER,
        hue="Strategy", palette="magma", jitter=True, alpha=0.6, legend=False
    )
    
    plt.xlabel("Scenario", fontsize=16)
    plt.ylabel("Beta Value (0=Green, 1=Cost)", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.ylim(-0.1, 1.1)  # Lock Y-axis since Beta is 0-1
    
    filename_beta = "SIablated_parameter_consistency_beta.png"
    plt.tight_layout()
    plt.savefig(filename_beta, dpi=300)
    print(f"✅ Saved Beta plot: {filename_beta}")
    plt.close()

if __name__ == "__main__":
    plot_parameters()