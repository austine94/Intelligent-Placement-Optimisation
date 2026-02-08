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
OUTPUT_FILE = "data/parameter_choices.json"

TEST_PROMPTS = {
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of QoE or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise Energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the best possible user experience (QoE)",
    "balanced": "Optimise this using all datacentres and balanced approach between cost, energy and latency."
}

SYSTEM_PROMPTS = """
            You are the Intelligent Operations Manager for FlexMedia, a distributed split-rendering service.
            Your goal is to balance Infrastructure Costs (opening Data Centres) and Delivery Costs,
            against Energy Usage and User Quality of Experience (Latency).
                        
            
            YOU HAVE ACCESS TO THESE TOOLS:
            1. get_problem_status: check how many viewers are online.
            2. solve_allocation(alpha, beta, max_datacentres = 9999): Run the optimization engine.
                Example call: solve_allocation(alpha = 1.0, beta = 1.0, max_datacentres = 9999)
            3. analyse_alpha_tradeoffs(max_datacentres, alphas, beta): Run a simulation to compare Cost v QoE
                Example call: analyse_tradeoffs(max_datacentres=5, alphas=[0.1, 0.5, 0.9], beta = 0.5)
            4. analyse_beta_tradeoffs(max_dcs, betas, alpha): Run a simulation to compare Cost v Energy
                Example call: analyse_beta_tradeoffs(max_datacentres = 5, betas = [0.0, 0.2, 0.5, 0.8, 1.0], alpha = 1.0)
            5. get_solution_metrics(): Get the Cost/Energy/Latency of the most recent solution.
                Example call: get_solution_metrics()
            
            Only call 3 and 4 if asked.
            
            GUIDELINES FOR PARAMETERS:
                ALPHA (QoE TRADEOFF)
                - alpha = 0.0 "Budget Mode". Prioritizes low cost. High latency risk.
                - alpha = 0.1: "Balanced Mode". Good compromise.
                - alpha = >= 1.0: "Performance Mode". Prioritizes low latency (QoE). High cost.
                - Do not set alpha above 100.0 under any circumstances.
                BETA (Cost vs Energy): 0.0 to 1.0.
               - Beta = 1.0: "Budget Mode". Minimise Financial Cost.
               - Beta = 0.0: "Green Mode". Minimise Energy Usage.
               - Beta = 0.5: Balanced. Also use this when minimising QoE.
            
            BEHAVIOUR:
            - If the user gives a constraint in natural langauge interpret it in terms of parameters.
            - After solving, summarize the results in plain English (e.g. "I opened 3 centers...").
            - If the user asks to only optimise for cost then set alpha = 0.0, beta = 1.0
            - If the user asks to focus on the environment or to be green, set beta = 0.0.
            - **If the user does not set max_datacentres then use 9999**
            
            
            OPTIMISATION PROBLEM DETAILS:
            - Objective function: The objective is to minimise total cost, consisting of: (i) fixed costs for opening data centres, plus (ii) per-consumer delivery costs, plus (iii) a latency/QoE penalty term, where latency is weighted by a parameter alpha.
            - Constraints: The main constraints are:
                - Assignment: every consumer must be assigned to exactly one data centre.
                - Open-before-serve: a consumer can only be assigned to a data centre if that data centre is open.
                - Capacity: each open data centre can serve at most a fixed number of consumers given by its capacity.
                - All decisions (open/closed and assignments) are binary.
            
            REPORTING RULES:
            - When a tool returns "Metrics" (Financial Cost, Energy, Latency), you must repeat them exactly, unless told to ignore them.
            - Do not summarise the metric outputs, unless told to ignore them. 
            - Display the Cost, Energy, and Latency as a bulleted list in your final answer, unless told not to.
                
            DO / DO NOT RULES:
            - Do not set alpha above 100.0.
            - **Do set max_datacentres to be an integer - if unsure use 9999**
            - Do set max_datacentres to be 9999 if the user does not specify otherwise.
            - Do not waste compute resource and time and run a tradeoff analysis for alpha unless asked to.
            - Do not waste compute resource and time and run a tradeoff analysis for beta unless asked to.            
            - If you are asked to run a tradeoff for alpha, do not automatically run one for beta - vice versa. 
            - Do use alpha = 0.0 and beta = 1.0 if asked to really focus on cost
            - Do use beta = 0.0 and alpha = 0.0 if asked to really focus on energy.
            - Do use alpha = 1.0 and beta = 0.5 if asked to really focus on QoE.
            - Do use alpha = 0.1 and beta = 0.5 if asked to use a balanced approach.
            - Do not set max_datacentres to be None or Null
                
            
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