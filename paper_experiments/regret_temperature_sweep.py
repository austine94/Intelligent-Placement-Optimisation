#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark objective regret across scenarios for a temperature sweep.

For each generated world seed and prompt scenario, this script:
1) solves the scenario with the fixed optimal parameters for that scenario;
2) asks the local agent to choose solve_allocation parameters at each temperature;
3) computes objective regret percentage relative to the fixed optimum; and
4) writes both raw JSON results and average-regret tables.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


# --- CONFIGURATION ---
N_REPS = 80
MODEL = "llama3.2"
TEMPERATURES = [0.0, 0.2, 0.5, 0.7, 1.0]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "regret_temperature_sweep_results.json"
REGRET_TABLE_CSV = EXPERIMENT_DATA_DIR / "regret_temperature_sweep_average_regret_table.csv"
REGRET_TABLE_MD = EXPERIMENT_DATA_DIR / "regret_temperature_sweep_average_regret_table.md"

SERVER_SCRIPT = PROJECT_ROOT / "server" / "mcp_server.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_scenario.py"

TEST_PROMPTS = {
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of latency or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the lowest possible latency and best user experience.",
    "tradeoff_cost_energy": "Optimise this using all datacentres. Balance financial cost and energy usage; latency is not the main priority.",
    "tradeoff_cost_latency": "Optimise this using all datacentres. Keep financial cost controlled, but maintain acceptable user experience.",
    "tradeoff_energy_latency": "Optimise this using all datacentres. Prefer a greener allocation, but do not let user experience become poor.",
}
SCENARIO_LABELS = {
    "focus_cost": "Cost",
    "focus_energy": "Energy",
    "focus_latency": "Latency",
    "tradeoff_cost_energy": "Cost-Energy",
    "tradeoff_cost_latency": "Cost-Latency",
    "tradeoff_energy_latency": "Energy-Latency",
}

SCENARIO_ORDER = [
    "focus_cost",
    "focus_energy",
    "focus_latency",
    "tradeoff_cost_energy",
    "tradeoff_cost_latency",
    "tradeoff_energy_latency",
]
# Fixed optimum parameters used as the regret baseline for each scenario.
OPTIMAL_PARAMS = {
    "focus_cost": {"alpha": 0.0, "beta": 1.0, "max_datacentres": 9999},
    "focus_energy": {"alpha": 0.0, "beta": 0.0, "max_datacentres": 9999},
    "focus_latency": {"alpha": 1.0, "beta": 0.5, "max_datacentres": 9999},
    "tradeoff_cost_energy": {"alpha": 0.0, "beta": 0.5, "max_datacentres": 9999},
    "tradeoff_cost_latency": {"alpha": 0.1, "beta": 1.0, "max_datacentres": 9999},
    "tradeoff_energy_latency": {"alpha": 0.1, "beta": 0.0, "max_datacentres": 9999},
}

def parse_metrics_from_text(text_output: str) -> dict[str, float]:
    """Extract Financial Cost / Energy Usage / Latency numeric fields from tool output."""
    metrics: dict[str, float] = {}

    cost_match = re.search(r"Financial Cost:\s*\$?([-+]?(?:\d+\.?\d*|\.\d+))", text_output)
    energy_match = re.search(r"Energy Usage:\s*([-+]?(?:\d+\.?\d*|\.\d+))", text_output)
    lat_match = re.search(r"Latency.*:\s*([-+]?(?:\d+\.?\d*|\.\d+))", text_output)

    if cost_match:
        metrics["cost"] = float(cost_match.group(1))
    if energy_match:
        metrics["energy"] = float(energy_match.group(1))
    if lat_match:
        metrics["latency"] = float(lat_match.group(1))

    return metrics


def parse_objective_from_text(text_output: str) -> float | None:
    """Extract the objective function value from solve_allocation output."""
    match = re.search(r"Objective Func:\s*([-+]?(?:\d+\.?\d*|\.\d+))", text_output)
    if not match:
        return None
    return float(match.group(1))


def compute_regret(agent_objective: float, optimal_objective: float) -> dict[str, float | None]:
    """Compute absolute and percentage objective regret from the optimum."""
    regret_absolute = agent_objective - optimal_objective
    regret_percent = None
    if optimal_objective != 0:
        regret_percent = (regret_absolute / optimal_objective) * 100.0

    return {
        "objective_regret_absolute": regret_absolute,
        "objective_regret_percent": regret_percent,
    }


def average(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None when no values are available."""
    if not values:
        return None
    return sum(values) / len(values)


def build_average_regret_table(results: dict[str, Any]) -> list[dict[str, str]]:
    """Create rows with scenarios on rows and temperatures on columns."""
    table_rows: list[dict[str, str]] = []

    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for temperature in TEMPERATURES:
            temp_key = str(temperature)
            runs = results.get(temp_key, {}).get(scenario, [])
            regrets = [
                float(run["objective_regret_percent"])
                for run in runs
                if run.get("objective_regret_percent") is not None
            ]
            avg_regret = average(regrets)
            row[temp_key] = "" if avg_regret is None else f"{avg_regret:.4f}"
        table_rows.append(row)

    return table_rows


def write_regret_table_csv(table_rows: list[dict[str, str]]) -> None:
    """Write the average regret percentage table as CSV."""
    fieldnames = ["scenario", *[str(temperature) for temperature in TEMPERATURES]]
    with REGRET_TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_regret_table_markdown(table_rows: list[dict[str, str]]) -> None:
    """Write the average regret percentage table as Markdown for quick viewing."""
    headers = ["Scenario", *[str(temperature) for temperature in TEMPERATURES]]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in table_rows:
        values = [row["scenario"], *[row[str(temperature)] for temperature in TEMPERATURES]]
        lines.append("| " + " | ".join(values) + " |")

    REGRET_TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_regret_table(table_rows: list[dict[str, str]]) -> None:
    """Print the average regret percentage table to stdout."""
    headers = ["Scenario", *[str(temperature) for temperature in TEMPERATURES]]
    rows = [[row["scenario"], *[row[str(temperature)] for temperature in TEMPERATURES]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print("\nAverage objective regret percentage from optimum:")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


async def call_solve_allocation(session: Any, arguments: dict[str, Any]) -> tuple[dict[str, float], float | None]:
    """Call solve_allocation and parse metrics plus objective value."""
    result = await session.call_tool("solve_allocation", arguments)
    output_text = result.content[0].text
    return parse_metrics_from_text(output_text), parse_objective_from_text(output_text)


async def run_regret_temperature_sweep() -> None:
    """Run regret benchmarking for every scenario and temperature."""
    import ollama
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    print(
        f"🚀 Regret temperature sweep: {N_REPS} reps x {len(TEMPERATURES)} temps x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Model: {MODEL} (Local)")
    print(f"🌡️ Temperatures: {TEMPERATURES}")
    print(f"🎯 Optimal parameters: {OPTIMAL_PARAMS}")

    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_SCRIPT)],
        env=os.environ.copy(),
    )

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(temperature): {scenario: [] for scenario in TEST_PROMPTS}
        for temperature in TEMPERATURES
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in mcp_tools.tools
            ]

            for rep in range(N_REPS):
                print(f"\n📢 --- REP {rep + 1}/{N_REPS} (World Seed {rep}) ---")
                subprocess.run(["python", str(GENERATOR_SCRIPT), str(rep)], check=True, capture_output=True)

                optimal_outputs: dict[str, dict[str, Any]] = {}
                for scenario in SCENARIO_ORDER:
                    try:
                        optimal_metrics, optimal_objective = await call_solve_allocation(
                            session,
                            OPTIMAL_PARAMS[scenario],
                        )
                        optimal_outputs[scenario] = {
                            "optimal_tool_args": OPTIMAL_PARAMS[scenario],
                            "optimal_metrics": optimal_metrics,
                            "optimal_objective": optimal_objective,
                        }
                    except Exception as exc:
                        optimal_outputs[scenario] = {
                            "optimal_tool_args": OPTIMAL_PARAMS[scenario],
                            "error": f"Optimal baseline error: {exc}",
                        }

                for temperature in TEMPERATURES:
                    print(f"  🌡️ Temp: {temperature}")
                    for scenario, prompt in TEST_PROMPTS.items():
                        print(f"    > Strat: {scenario:<15}...", end="", flush=True)

                        record: dict[str, Any] = {
                            "rep": rep,
                            "seed": rep,
                            "temperature": temperature,
                            "prompt_category": scenario,
                            **optimal_outputs[scenario],
                        }

                        if record.get("optimal_objective") is None:
                            record["error"] = record.get("error", "Could not parse optimal objective")
                            results_db[str(temperature)][scenario].append(record)
                            print(" ❌ Baseline failed")
                            continue

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
                                options={"seed": rep, "temperature": temperature},
                            )

                            tool_calls = response["message"].get("tool_calls", [])
                            if not tool_calls:
                                record["error"] = "No tool call from agent"
                                results_db[str(temperature)][scenario].append(record)
                                print(" ❌ Agent failed (No Tool)")
                                continue

                            tool_call = tool_calls[0]
                            agent_fn_name = tool_call["function"]["name"]
                            agent_fn_args = tool_call["function"]["arguments"]
                            record["agent_tool_args"] = agent_fn_args

                            if agent_fn_name != "solve_allocation":
                                record["error"] = f"Agent called unexpected tool: {agent_fn_name}"
                                results_db[str(temperature)][scenario].append(record)
                                print(f" ⚠️ Agent skipped ({agent_fn_name})")
                                continue

                            agent_metrics, agent_objective = await call_solve_allocation(
                                session,
                                agent_fn_args,
                            )
                            record["agent_metrics"] = agent_metrics
                            record["agent_objective"] = agent_objective

                            if agent_objective is None:
                                record["error"] = "Could not parse agent objective"
                                results_db[str(temperature)][scenario].append(record)
                                print(" ⚠️ Missing objective")
                                continue

                            record.update(compute_regret(agent_objective, record["optimal_objective"]))
                            results_db[str(temperature)][scenario].append(record)
                            print(" ✅")

                        except Exception as exc:
                            record["error"] = f"Agent run error: {exc}"
                            results_db[str(temperature)][scenario].append(record)
                            print(f" ❌ Agent error: {exc}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_average_regret_table(results_db)
    write_regret_table_csv(table_rows)
    write_regret_table_markdown(table_rows)
    print_regret_table(table_rows)

    print(f"\n✅ Regret temperature sweep complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Average regret table CSV saved to {REGRET_TABLE_CSV}")
    print(f"✅ Average regret table Markdown saved to {REGRET_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_regret_temperature_sweep())
