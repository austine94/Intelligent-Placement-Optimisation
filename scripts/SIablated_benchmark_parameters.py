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
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of QoE or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise Energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the best possible user experience (QoE)",
    "balanced": "Optimise this using all datacentres and balanced approach between cost, energy and latency."
}

SYSTEM_PROMPTS = """
            
            """

#manually define the tool since we aren't asking the MCP server and just checking model
TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "solve_allocation",
            "description": "Solves the facility location problem. Alpha weights Latency (QoE). Beta weights Cost vs Energy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {
                        "type": "number",
                        "description": "Weight for Latency/QoE. 0.0=Ignore Latency, 100.0=Critical."
                    },
                    "beta": {
                        "type": "number",
                        "description": "Tradeoff between Financial Cost (1.0) and Energy (0.0). 0.5 is balanced."
                    },
                    "max_datacentres": {
                        "type": "integer",
                        "description": "Max number of DCs to open."
                    }
                },
                "required": ["alpha", "beta"]
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