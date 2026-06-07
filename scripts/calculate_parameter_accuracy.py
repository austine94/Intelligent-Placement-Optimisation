#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:45:10 2026

@author: localadmin
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# --- CONFIGURATION ---
INPUT_FILE = "data/parameter_choices.json"
TOLERANCE = 0.05  # Allow small deviations (e.g. 0.0 is close enough to 0.001)

# --- GROUND TRUTH RULES ---
# These match the explicit logic you provided in the prompt.
TARGETS = {
    "focus_cost": {"alpha": 0.0, "beta": 1.0},
    "focus_energy": {"alpha": 0.0, "beta": 0.0},
    "focus_latency": {"alpha": 1.0, "beta": 0.5},
    "tradeoff_cost_energy": {"alpha": 0.0, "beta": 0.5},
    "tradeoff_cost_latency": {"alpha": 0.1, "beta": 1.0},
    "tradeoff_energy_latency": {"alpha": 0.1, "beta": 0.0},
}

# Clean labels for the plot
CATEGORY_MAP = {
    "focus_cost": "Cost",
    "focus_energy": "Energy",
    "focus_latency": "Latency",
    "tradeoff_cost_energy": "Cost-Energy",
    "tradeoff_cost_latency": "Cost-Latency",
    "tradeoff_energy_latency": "Energy-Latency",
}
ORDER = ["Cost", "Energy", "Latency", "Cost-Energy", "Cost-Latency", "Energy-Latency"]

def is_correct(actual, target):
    """Checks if the value matches the target within tolerance."""
    if actual is None:
        return False
    try:
        val = float(actual)
        return math.isclose(val, target, abs_tol=TOLERANCE)
    except (ValueError, TypeError):
        return False

def calculate_and_plot_accuracy():
    if not os.path.exists(INPUT_FILE):
        print("❌ File not found. Run benchmark_parameters.py first!")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    results = []

    # 1. Score the data
    for category, runs in data.items():
        if category not in TARGETS:
            continue # Skip if we don't have a rule for it

        target = TARGETS[category]
        
        for run in runs:
            # Check Alpha
            alpha_val = run.get('alpha')
            alpha_ok = is_correct(alpha_val, target['alpha'])
            
            # Check Beta
            beta_val = run.get('beta')
            beta_ok = is_correct(beta_val, target['beta'])
            
            results.append({
                "Strategy": CATEGORY_MAP.get(category, category),
                "Alpha Correct": 1 if alpha_ok else 0,
                "Beta Correct": 1 if beta_ok else 0
            })

    if not results:
        print("❌ No data to process.")
        return

    df = pd.DataFrame(results)

    # 2. Calculate Proportions (Means)
    # This gives us a value 0.0 to 1.0 (e.g. 0.95 = 95%)
    summary = df.groupby("Strategy")[["Alpha Correct", "Beta Correct"]].mean().reset_index()

    # Print Text Summary
    print("\n--- ✅ Accuracy Report (Proportion Correct) ---")
    print(summary.round(2))
    print("-" * 40)

    # 3. Plotting
    sns.set_theme(style="whitegrid")
    
    # --- PLOT 1: ALPHA ACCURACY ---
    plt.figure(figsize=(6, 5))
    sns.barplot(
        data=summary, x="Strategy", y="Alpha Correct", order=ORDER,
        color="#1f77b4" # Blue
    )
    plt.title("Adherence to Alpha Rules", fontsize=14, fontweight='bold')
    plt.ylabel("Proportion Correct (0-1)", fontsize=12)
    plt.xlabel("Strategy", fontsize=12)
    plt.ylim(0, 1.05) # Lock scale 0 to 100%
    
    # Add percentage labels on top of bars
    for index, row in summary.iterrows():
        # Find x location based on order
        try:
            x_loc = ORDER.index(row["Strategy"])
            plt.text(x_loc, row["Alpha Correct"] + 0.02, f"{row['Alpha Correct']:.0%}", 
                     ha='center', color='black', fontweight='bold')
        except ValueError:
            pass

    plt.tight_layout()
    plt.savefig("accuracy_alpha.png", dpi=300)
    print("✅ Saved plot: accuracy_alpha.png")
    plt.close()

    # --- PLOT 2: BETA ACCURACY ---
    plt.figure(figsize=(6, 5))
    sns.barplot(
        data=summary, x="Strategy", y="Beta Correct", order=ORDER,
        color="#2ca02c" # Green
    )
    plt.title("Adherence to Beta Rules", fontsize=14, fontweight='bold')
    plt.ylabel("Proportion Correct (0-1)", fontsize=12)
    plt.xlabel("Strategy", fontsize=12)
    plt.ylim(0, 1.05)

    for index, row in summary.iterrows():
        try:
            x_loc = ORDER.index(row["Strategy"])
            plt.text(x_loc, row["Beta Correct"] + 0.02, f"{row['Beta Correct']:.0%}", 
                     ha='center', color='black', fontweight='bold')
        except ValueError:
            pass

    plt.tight_layout()
    plt.savefig("accuracy_beta.png", dpi=300)
    print("✅ Saved plot: accuracy_beta.png")
    plt.close()

if __name__ == "__main__":
    calculate_and_plot_accuracy()