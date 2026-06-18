#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark agent regret against a deterministic baseline solve.

For each generated world seed and prompt scenario:
1) Run the agent-style tool selection via ollama + MCP tools.
2) Run a deterministic baseline solve with fixed target parameters for that scenario.
3) Compute metric-wise absolute percentage regret for cost, energy, and latency.
"""

import asyncio
import json
import os
import re
import subprocess
from typing import Any, Dict

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- CONFIGURATION ---
N_REPS = 250
MODEL = "interstellarninja/llama3.1-8b-tools:latest"
TEMPERATURE = 0.7
RESULTS_FILE = "data/benchmark_regret_results.json"

TEST_PROMPTS = {
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of latency or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the lowest possible latency and best user experience.",
    "tradeoff_cost_energy": "Optimise this using all datacentres. Balance financial cost and energy usage; latency is not the main priority.",
    "tradeoff_cost_latency": "Optimise this using all datacentres. Keep financial cost controlled, but maintain acceptable user experience.",
    "tradeoff_energy_latency": "Optimise this using all datacentres. Prefer a greener allocation minimising energy, but do not let user experience become poor.",
}
DETERMINISTIC_TARGET_PARAMS = {
    "focus_cost": {"alpha": 0.0, "beta": 1.0, "max_datacentres": 9999},
    "focus_energy": {"alpha": 0.0, "beta": 0.0, "max_datacentres": 9999},
    "focus_latency": {"alpha": 1.0, "beta": 0.5, "max_datacentres": 9999},
    "tradeoff_cost_energy": {"alpha": 0.0, "beta": 0.5, "max_datacentres": 9999},
    "tradeoff_cost_latency": {"alpha": 0.1, "beta": 1.0, "max_datacentres": 9999},
    "tradeoff_energy_latency": {"alpha": 0.1, "beta": 0.0, "max_datacentres": 9999},
}
# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
server_script = os.path.join(project_root, "server", "mcp_server.py")
generator_script = os.path.join(project_root, "scripts", "generate_scenario.py")


def parse_metrics_from_text(text_output: str) -> Dict[str, float]:
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


def parse_objective_from_text(text_output: str) -> float | None:
    match = re.search(r"Objective Func:\s*([\d\.]+)", text_output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


METRIC_REGRET_KEYS = ("cost", "energy", "latency")


def metric_absolute_percentage_regret(
    agent_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    metric: str,
) -> float | None:
    agent_value = agent_metrics.get(metric)
    baseline_value = baseline_metrics.get(metric)
    if agent_value is None or baseline_value is None or baseline_value == 0:
        return None
    return (abs(agent_value - baseline_value) / abs(baseline_value)) * 100.0


def compute_regret(
    agent_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    agent_obj: float | None = None,
    baseline_obj: float | None = None,
) -> Dict[str, float | None]:
    metric_regrets = {
        f"{metric}_regret_percent": metric_absolute_percentage_regret(
            agent_metrics,
            baseline_metrics,
            metric,
        )
        for metric in METRIC_REGRET_KEYS
    }
    available_metric_regrets = [value for value in metric_regrets.values() if value is not None]
    metric_regret_percent = (
        sum(available_metric_regrets) / len(available_metric_regrets)
        if len(available_metric_regrets) == len(METRIC_REGRET_KEYS)
        else None
    )

    objective_regret_absolute = None
    objective_regret_percent = None
    if agent_obj is not None and baseline_obj is not None:
        objective_regret_absolute = agent_obj - baseline_obj
        if baseline_obj != 0:
            objective_regret_percent = (abs(objective_regret_absolute) / abs(baseline_obj)) * 100.0

    return {
        **metric_regrets,
        "metric_regret_percent": metric_regret_percent,
        "objective_regret_absolute": objective_regret_absolute,
        "objective_regret_percent": objective_regret_percent,
    }


async def run_regret_benchmark() -> None:
    print(f" Regret Benchmark: {N_REPS} reps x {len(TEST_PROMPTS)} prompts")
    print(f" Model: {MODEL} (Local)")
    print(f" Agent Temperature: {TEMPERATURE}")

    server_params = StdioServerParameters(command="python", args=[server_script], env=os.environ.copy())

    results_db = {k: [] for k in TEST_PROMPTS.keys()}

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
                subprocess.run(["python", generator_script, str(rep)], check=True, capture_output=True)

                for category, prompt in TEST_PROMPTS.items():
                    print(f"  > Strat: {category:<15}...", end="", flush=True)

                    record: Dict[str, Any] = {
                        "rep": rep,
                        "seed": rep,
                        "prompt_category": category,
                        "agent_temperature": TEMPERATURE,
                    }

                    # Agent run
                    try:
                        messages = [
                            {"role": "system", "content": "You are the FlexMedia Ops Manager. Use 'solve_allocation' to optimize."},
                            {"role": "user", "content": prompt},
                        ]
                        response = ollama.chat(
                            model=MODEL,
                            messages=messages,
                            tools=ollama_tools,
                            options={"seed": rep, "temperature": TEMPERATURE},
                        )

                        tool_calls = response["message"].get("tool_calls", [])
                        if not tool_calls:
                            record["error"] = "No tool call from agent"
                            results_db[category].append(record)
                            print(" ERROR Agent failed (No Tool)")
                            continue

                        tool_call = tool_calls[0]
                        agent_fn_name = tool_call["function"]["name"]
                        agent_fn_args = tool_call["function"]["arguments"]
                        record["agent_tool_args"] = agent_fn_args

                        if agent_fn_name != "solve_allocation":
                            record["error"] = f"Agent called unexpected tool: {agent_fn_name}"
                            results_db[category].append(record)
                            print(f" WARNING Agent skipped ({agent_fn_name})")
                            continue

                        agent_result = await session.call_tool(agent_fn_name, agent_fn_args)
                        agent_output = agent_result.content[0].text
                        record["agent_metrics"] = parse_metrics_from_text(agent_output)
                        record["agent_objective"] = parse_objective_from_text(agent_output)

                    except Exception as exc:
                        record["error"] = f"Agent run error: {exc}"
                        results_db[category].append(record)
                        print(f" ERROR Agent error: {exc}")
                        continue

                    # Deterministic baseline run
                    baseline_args = DETERMINISTIC_TARGET_PARAMS[category]
                    record["baseline_tool_args"] = baseline_args
                    try:
                        baseline_result = await session.call_tool("solve_allocation", baseline_args)
                        baseline_output = baseline_result.content[0].text
                        record["baseline_metrics"] = parse_metrics_from_text(baseline_output)
                        record["baseline_objective"] = parse_objective_from_text(baseline_output)
                    except Exception as exc:
                        record["error"] = f"Baseline run error: {exc}"
                        results_db[category].append(record)
                        print(f" ERROR Baseline error: {exc}")
                        continue

                    agent_obj = record.get("agent_objective")
                    baseline_obj = record.get("baseline_objective")

                    if agent_obj is None or baseline_obj is None:
                        record["error"] = "Could not parse objective(s) for regret calculation"
                        results_db[category].append(record)
                        print(" WARNING Missing objective")
                        continue

                    record.update(
                        compute_regret(
                            record["agent_metrics"],
                            record["baseline_metrics"],
                            agent_obj,
                            baseline_obj,
                        )
                    )
                    results_db[category].append(record)
                    print(" OK")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    print(f"\nOK Regret benchmark complete. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(run_regret_benchmark())
