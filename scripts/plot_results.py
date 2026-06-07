import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
RESULTS_FILE = "data/benchmark_results.json"

#mapping for JSON to x labels
CATEGORY_MAP = {
    "focus_cost": "Cost",
    "focus_energy": "Energy",
    "focus_latency": "Latency",
    "tradeoff_cost_energy": "Cost-Energy",
    "tradeoff_cost_latency": "Cost-Latency",
    "tradeoff_energy_latency": "Energy-Latency",
}

def load_and_process_data():
    if not os.path.exists(RESULTS_FILE):
        print(f"File not found: {RESULTS_FILE}")
        return None

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    rows = []
    skipped_count = 0
    total_count = 0

    #Process only valid runs
    for category, runs in data.items():
        for run in runs:
            total_count += 1
            metrics = run.get('metrics', {})
            
            if not metrics or 'cost' not in metrics:
                skipped_count += 1
                continue
            
            clean_cat = CATEGORY_MAP.get(category, category)
                
            row = {
                "Category": clean_cat,
                "Cost ($)": metrics.get('cost'),
                "Energy (Units)": metrics.get('energy'),
                "Latency (ms)": metrics.get('latency')
            }
            rows.append(row)

    print(f"Processed {len(rows)} valid runs.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count}/{total_count} runs due to missing metrics.")

    return pd.DataFrame(rows)

def plot_benchmarks(df):
    # Set a clean visual style
    sns.set_theme(style="whitegrid")
    
    #set metrics and their specific colors/filenames
    metrics_config = [
        {"col": "Cost ($)",       "color": "#1f77b4", "filename": "benchmark_cost.png"},
        {"col": "Energy (Units)", "color": "#2ca02c", "filename": "benchmark_energy.png"},
        {"col": "Latency (ms)",   "color": "#d62728", "filename": "benchmark_latency.png"}
    ]
    
    #define the specific order we want on the x axis
    order = ["Cost", "Energy", "Latency", "Cost-Energy", "Cost-Latency", "Energy-Latency"]
    
    for config in metrics_config:
        metric = config["col"]
        
        plt.figure(figsize=(6, 5))
        
        sns.barplot(
            data=df, 
            x="Category", 
            y=metric, 
            order=order,
            color=config["color"], 
            capsize=.1, 
            errorbar='sd' 
        )
        
        plt.title(f"Average {metric}", fontsize=14, fontweight='bold')
        plt.xlabel("Scenario", fontsize=12)
        plt.ylabel(metric, fontsize=11)
        
        plt.tight_layout()
        
        plt.savefig(config["filename"], dpi=300)
        print(f"✅ Saved plot: {config['filename']}")
        plt.close()

if __name__ == "__main__":
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        plot_benchmarks(df)
    else:
        print("No valid data found to plot.")