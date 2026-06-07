#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:02:52 2026

@author: localadmin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:52:46 2026

@author: localadmin
"""

import ollama
import json
import os
import tqdm  #for progress bar

# --- CONFIGURATION ---
N_REPS = 100
MODEL = "llama3.2"
OUTPUT_FILE = "data/SIablated_parameter_choices.json"

TEST_PROMPTS = {
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of latency or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the lowest possible latency and best user experience.",
    "tradeoff_cost_energy": "Optimise this using all datacentres. Balance financial cost and energy usage; latency is not the main priority.",
    "tradeoff_cost_latency": "Optimise this using all datacentres. Keep financial cost controlled, but maintain acceptable user experience.",
    "tradeoff_energy_latency": "Optimise this using all datacentres. Prefer a greener allocation, but do not let user experience become poor.",
}
SYSTEM_PROMPTS = """
You are the Intelligent Operations Manager for FlexMedia, a distributed split-rendering service.
Your goal is to choose optimisation parameters for infrastructure cost, delivery cost, energy usage,
and user latency.

YOU HAVE ACCESS TO THESE TOOLS:
1. get_problem_status: check how many viewers are online.
2. solve_allocation(alpha, beta, max_datacentres = 9999): Run the optimisation engine.
   Example call: solve_allocation(alpha = 1.0, beta = 1.0, max_datacentres = 9999)
3. analyse_alpha_tradeoffs(max_datacentres, alphas, beta): compare cost/energy outcomes across latency weights.
4. analyse_beta_tradeoffs(max_datacentres, betas, alpha): compare financial cost vs energy weights.
5. get_solution_metrics(): Get the Cost/Energy/Latency of the most recent solution.

Only call analyse_alpha_tradeoffs or analyse_beta_tradeoffs if asked.

PARAMETER MEANINGS:
ALPHA (latency trade-off)
- alpha = 0.0: ignore latency in the objective.
- alpha = 0.1: moderate latency pressure for trade-off scenarios.
- alpha >= 1.0: strong latency/performance priority.
- Do not set alpha above 100.0 under any circumstances.
BETA (financial cost vs energy)
- beta = 1.0: prioritise financial cost.
- beta = 0.0: prioritise energy usage.
- beta = 0.5: trade off financial cost and energy equally.

CANONICAL SCENARIO RULES:
- Cost only: set alpha = 0.0, beta = 1.0.
- Energy only: set alpha = 0.0, beta = 0.0.
- Latency / user-experience only: set alpha = 1.0, beta = 0.5.
- Cost-energy trade-off: set alpha = 0.0, beta = 0.5.
- Cost-latency trade-off: set alpha = 0.1, beta = 1.0.
- Energy-latency trade-off: set alpha = 0.1, beta = 0.0.
- If the user does not set max_datacentres then use 9999.

OPTIMISATION PROBLEM DETAILS:
- Objective function: minimise fixed data-centre opening costs, weighted per-consumer financial/energy delivery costs, and a latency penalty weighted by alpha.
- Constraints: each consumer is assigned exactly once; consumers can only use open data centres; each open data centre has finite capacity; open/closed and assignment decisions are binary.

REPORTING RULES:
- When a tool returns Metrics (Financial Cost, Energy, Latency), repeat them exactly unless told to ignore them.
- Display Cost, Energy, and Latency as a bulleted list in your final answer unless told not to.

DO / DO NOT RULES:
- Do not set alpha above 100.0.
- Do set max_datacentres to be an integer; if unsure use 9999.
- Do not set max_datacentres to be None or Null.
- Do not run a trade-off analysis for alpha or beta unless asked.
- If asked to run a trade-off for alpha, do not automatically run one for beta, and vice versa.
"""

#manually define the tool since we aren't asking the MCP server and just checking model
TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "solve_allocation",
            "description": "Solves the facility location problem. Alpha weights latency. Beta weights financial cost vs energy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {
                        "type": "number",
                        "description": "Weight for latency. 0.0=ignore latency, 0.1=moderate trade-off, 1.0=latency priority."
                    },
                    "beta": {
                        "type": "number",
                        "description": "Financial cost vs energy. 1.0=cost priority, 0.0=energy priority, 0.5=cost-energy trade-off."
                    },
                    "max_datacentres": {
                        "type": "integer",
                        "description": "Max number of DCs to open. Use 9999 when unconstrained."
                    }
                },
                "required": ["alpha", "beta", "max_datacentres"]
            }
        }
    }
]

def run_parameter_benchmark():
    print(f"Benchmarking Parameter Selection ({N_REPS} reps per prompt)...")
    results = {k: [] for k in TEST_PROMPTS.keys()}

    #test each strategy
    for category, prompt in TEST_PROMPTS.items():
        print(f"\n--- Testing: {category} ---")
        
        # Simple loop (using tqdm for a progress bar if installed, else just range)
        iterator = range(N_REPS)
        try:
            import tqdm
            iterator = tqdm.tqdm(range(N_REPS), desc=category)
        except ImportError:
            pass

        for i in iterator:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS},
                {"role": "user", "content": prompt}
            ]

            try:
                #direct Ollama call
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITION,
                    options={
                        'seed': i,
                        'temperature': 0.7 
                    }
                )
                
                tool_calls = response['message'].get('tool_calls', [])
                
                if tool_calls:
                    args = tool_calls[0]['function']['arguments']
                    
                    # Capture the decision
                    record = {
                        "rep": i,
                        "alpha": args.get('alpha'),
                        "beta": args.get('beta')
                    }
                    results[category].append(record)
                else:
                    # Logic failed (LLM chatted instead of calling tool)
                    results[category].append({"rep": i, "error": "No tool call"})

            except Exception as e:
                print(f"Error: {e}")

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_parameter_benchmark()