# server/mcp_server.py
from fastmcp import FastMCP
import json
import numpy as np
import os
import sys

# 1. SETUP PATHS (Absolute paths prevent confusion)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
engine_path = os.path.join(project_root, 'optimisation_engine')
sys.path.append(engine_path)

from flexmedia_delivery_instance import flexmedia_delivery_instance

# 2. START SERVER
mcp = FastMCP("FlexMedia Optimizer")

# 3. DEFINE DATA PATH (Use project_root to be safe)
DATA_PATH = os.path.join(project_root, "data", "current_problem.json")

def load_problem_into_instance():
    """
    Function to load in JSON of problem instance and then write into 
    correct class structure.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No problem found at {DATA_PATH}. Run scripts/generate_scenario.py first.")

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # Instantiate the class (ignore random data)
    inst = flexmedia_delivery_instance(
        n_datacentres=data['n_datacentres'], 
        n_consumers=data['n_consumers'], 
        n_hotspots=1 
    )

    # Overwrite with data from JSON
    inst.datacentre_costs = np.array(data['datacentre_costs'])
    inst.datacentre_capacities = np.array(data['datacentre_capacities'])
    inst.consumer_costs = np.array(data['consumer_costs'])
    inst.consumer_latencies = np.array(data['consumer_latencies'])
    inst.datacentre_locations = np.array(data['datacentre_locations'])
    inst.consumer_locations = np.array(data['consumer_locations'])
    
    return inst

@mcp.tool()
def get_problem_status() -> str:
    """
    Returns the number of viewers and available datacentres.
    """
    if not os.path.exists(DATA_PATH):
        return "No problem data found."
    
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        
    return f"Current Scenario: {data['n_consumers']} Viewers, {data['n_datacentres']} Data Centres available."

@mcp.tool()
def solve_allocation(alpha: float, max_datacentres: int) -> str: 
    """
    Solves the facility location problem exactly.
    
    Args:
        alpha: Float to penalise latency (QoE) in objective.
        max_datacentres: Maximum number of facilities to open.
    """
    try:
        inst = load_problem_into_instance()
        
        # Solve
        inst.exact_solve(alpha=alpha)
        
        if inst.exact_obj is None:
            return "Solver failed or found no feasible solution."

        num_open = len(inst.open_centres)
        
        warning = ""
        if num_open > max_datacentres:
            warning = f"\nWARNING: Solution used {num_open} DCs, exceeding limit of {max_datacentres}."

        return (
            f"Optimization Successful.\n"
            f"Tradeoff Alpha: {alpha}\n"
            f"Total Objective: {inst.exact_obj:.2f}\n"
            f"Data Centres Opened: {num_open} (IDs: {inst.open_centres})\n"
            f"{warning}"
        )

    except Exception as e:
        return f"Error running optimization: {str(e)}"
    
    
if __name__ == "__main__":
    # Optional: simple log to prove it starts
    print("Starting FlexMedia Optimizer MCP server on stdio...", file=sys.stderr)
    mcp.run(transport="stdio")

    
    
