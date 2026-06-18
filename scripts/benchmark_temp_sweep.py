#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark system performance across prompt scenarios for a sweep of temperatures.

This script is intentionally separate from existing benchmark scripts.
It runs the real MCP tool call (solve_allocation) and records resulting metrics.
"""

import asyncio
import json
import os
import re
import subprocess
from typing import Any, Dict, List

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- CONFIGURATION ---
N_REPS = 250
MODEL = "interstellarninja/llama3.1-8b-tools:latest"
TEMPERATURES = [0.0, 0.25, 0.5, 0.75, 1.0]
RESULTS_FILE = "data/benchmark_temp_sweep_results.json"

TEST_PROMPTS = {
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of latency or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the lowest possible latency and best user experience.",
    "tradeoff_cost_energy": "Optimise this using all datacentres. Balance financial cost and energy usage; latency is not the main priority.",
    "tradeoff_cost_latency": "Optimise this using all datacentres. Keep financial cost controlled, but maintain acceptable user experience.",
    "tradeoff_energy_latency": "Optimise this using all datacentres. Prefer a greener allocation minimising energy, but do not let user experience become poor.",
}
# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
server_script = os.path.join(project_root, "server", "mcp_server.py")
generator_script = os.path.join(project_root, "scripts", "generate_scenario.py")


def parse_metrics_from_text(text_output: str) -> Dict[str, float]:
    """Extract Financial Cost / Energy Usage / Latency numeric fields from tool output."""
    metrics: Dict[str, float] = {}

    cost_match = re.search(r"Financial Cost:\s*\$?([\d\.]+)", text_output)
    energy_match = re.search(r"Energy Usage:\s*([\d\.]+)", text_output)
    lat_match = re.search(r"Latency.*:\s*([\d\.]+)", text_output)

    if cost_match:
        metrics["cost"] = float(cost_match.group(1))
    if energy_match:
        metrics["energy"] = float(energy_match.group(1))
    if lat_match:
        metrics["latency"] = float(lat_match.group(1))

    return metrics


async def run_temp_sweep_benchmark() -> None:
    print(
        f" Temp-Sweep Benchmark: {N_REPS} reps x {len(TEMPERATURES)} temps x {len(TEST_PROMPTS)} prompts"
    )
    print(f" Model: {MODEL} (Local)")
    print(f" Temperatures: {TEMPERATURES}")

    server_params = StdioServerParameters(
        command="python",
        args=[server_script],
        env=os.environ.copy(),
    )

    # Keep results grouped by temperature then category.
    results_db: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        str(temp): {category: [] for category in TEST_PROMPTS.keys()}
        for temp in TEMPERATURES
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            ollama_tools = []
            for tool in mcp_tools.tools:
                ollama_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )

            for rep in range(N_REPS):
                print(f"\n --- REP {rep + 1}/{N_REPS} (World Seed {rep}) ---")

                # Generate one world per rep so all temps/prompts share the same instance.
                subprocess.run(
                    ["python", generator_script, str(rep)],
                    check=True,
                    capture_output=True,
                )

                for temp in TEMPERATURES:
                    print(f"   Temp: {temp}")
                    for category, prompt in TEST_PROMPTS.items():
                        print(f"    > Strat: {category:<15}...", end="", flush=True)

                        messages = [
                            {
                                "role": "system",
                                "content": "You are the FlexMedia Ops Manager. Use 'solve_allocation' to optimize.",
                            },
                            {"role": "user", "content": prompt},
                        ]

                        try:
                            response = ollama.chat(
                                model=MODEL,
                                messages=messages,
                                tools=ollama_tools,
                                options={"seed": rep, "temperature": temp},
                            )

                            msg = response["message"]
                            tool_calls = msg.get("tool_calls", [])

                            if not tool_calls:
                                results_db[str(temp)][category].append(
                                    {
                                        "rep": rep,
                                        "seed": rep,
                                        "temperature": temp,
                                        "prompt_category": category,
                                        "error": "No tool call",
                                    }
                                )
                                print(" ERROR Failed (No Tool)")
                                continue

                            tool_call = tool_calls[0]
                            fn_name = tool_call["function"]["name"]
                            fn_args = tool_call["function"]["arguments"]

                            if fn_name != "solve_allocation":
                                results_db[str(temp)][category].append(
                                    {
                                        "rep": rep,
                                        "seed": rep,
                                        "temperature": temp,
                                        "prompt_category": category,
                                        "tool_args": fn_args,
                                        "error": f"Unexpected tool: {fn_name}",
                                    }
                                )
                                print(f" WARNING Skipped ({fn_name})")
                                continue

                            result = await session.call_tool(fn_name, fn_args)
                            output_text = result.content[0].text
                            parsed = parse_metrics_from_text(output_text)

                            results_db[str(temp)][category].append(
                                {
                                    "rep": rep,
                                    "seed": rep,
                                    "temperature": temp,
                                    "prompt_category": category,
                                    "tool_args": fn_args,
                                    "metrics": parsed,
                                }
                            )
                            print(" OK")

                        except Exception as exc:
                            results_db[str(temp)][category].append(
                                {
                                    "rep": rep,
                                    "seed": rep,
                                    "temperature": temp,
                                    "prompt_category": category,
                                    "error": str(exc),
                                }
                            )
                            print(f" ERROR Error: {exc}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    print(f"\nOK Temp-sweep benchmark complete. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run_temp_sweep_benchmark())
