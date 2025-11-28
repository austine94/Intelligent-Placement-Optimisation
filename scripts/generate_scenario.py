#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 18:26:59 2025

@author: localadmin
"""

#scripts/generate_scenario.py
#This is a script to generate new optimisation instances

import json
import numpy as np
import os
import sys

#amend paths for importing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
engine_path = os.path.join(project_root, 'optimisation_engine')
sys.path.append(engine_path)

from flexmedia_delivery_instance import flexmedia_delivery_instance


def generate_and_save():
    '''
    This function generates a new optimisation instance and saves it
    as a JSON in the desired form. This can then be accessed by the agent
    '''
    print("Generating new optimisation scenario...")
    
    #Create the optimisation instance - these are just default values 
    inst = flexmedia_delivery_instance(
        n_datacentres=5, 
        n_consumers=500, 
        n_hotspots=2,
        edge_capacity = 500,
        core_capacity = 500
    )
    
    #Extract data into dict structure
    problem_data = {
        "n_datacentres": len(inst.datacentre_costs),
        "n_consumers": len(inst.consumer_costs[0]),
        # We save the arrays required for the solver
        "datacentre_costs": inst.datacentre_costs.tolist(),
        "datacentre_capacities": inst.datacentre_capacities.tolist(),
        "consumer_costs": inst.consumer_costs.tolist(),
        "consumer_latencies": inst.consumer_latencies.tolist(),
        # We save locations just for visualization/debugging later
        "datacentre_locations": inst.datacentre_locations.tolist(),
        "consumer_locations": inst.consumer_locations.tolist()
    }
    
    #Save dict as json
    output_path = os.path.join("data", "current_problem.json")
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(problem_data, f, indent=2)
        
    print(f"Scenario saved to {output_path}")

if __name__ == "__main__":
    generate_and_save()