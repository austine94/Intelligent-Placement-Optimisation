#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark objective regret across scenarios for a local SLM model sweep.

For each generated world seed and prompt scenario, this script:
1) solves the scenario with the fixed optimal parameters for that scenario;
2) asks each configured local Ollama model to choose solve_allocation parameters;
3) computes objective regret percentage relative to the fixed optimum; and
4) writes both raw JSON results and average-regret tables.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from regret_temperature_sweep import (
    OPTIMAL_PARAMS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    TEST_PROMPTS,
    average,
    call_solve_allocation,
    compute_regret,
)


# --- CONFIGURATION ---
N_REPS = 80
TEMPERATURE = 0.0
MODELS = [
    "llama3.2",
    "llama3.2:1b",
    "llama3.1:8b",
    "qwen3:4b",
    "gemma3:4b",
]

PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "regret_model_sweep_results.json"
REGRET_TABLE_CSV = EXPERIMENT_DATA_DIR / "regret_model_sweep_average_regret_table.csv"
REGRET_TABLE_MD = EXPERIMENT_DATA_DIR / "regret_model_sweep_average_regret_table.md"

SERVER_SCRIPT = PROJECT_ROOT / "server" / "mcp_server.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_scenario.py"


def build_average_regret_table(results: dict[str, Any]) -> list[dict[str, str]]:
    """Create rows with scenarios on rows and model names on columns."""
    table_rows: list[dict[str, str]] = []

    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for model in MODELS:
            runs = results.get(model, {}).get(scenario, [])
            regrets = [
                float(run["objective_regret_percent"])
                for run in runs
                if run.get("objective_regret_percent") is not None
            ]
            avg_regret = average(regrets)
            row[model] = "" if avg_regret is None else f"{avg_regret:.4f}"
        table_rows.append(row)

    return table_rows


def write_regret_table_csv(table_rows: list[dict[str, str]]) -> None:
    """Write the average regret percentage table as CSV."""
    fieldnames = ["scenario", *MODELS]
    with REGRET_TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_regret_table_markdown(table_rows: list[dict[str, str]]) -> None:
    """Write the average regret percentage table as Markdown for quick viewing."""
    headers = ["Scenario", *MODELS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in table_rows:
        values = [row["scenario"], *[row[model] for model in MODELS]]
        lines.append("| " + " | ".join(values) + " |")

    REGRET_TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_regret_table(table_rows: list[dict[str, str]]) -> None:
    """Print the average regret percentage table to stdout."""
    headers = ["Scenario", *MODELS]
    rows = [[row["scenario"], *[row[model] for model in MODELS]] for row in table_rows]
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


async def run_regret_model_sweep() -> None:
    """Run regret benchmarking for every scenario and configured Ollama model."""
    import ollama
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print(
        f"🚀 Regret model sweep: {N_REPS} reps x {len(MODELS)} models x {len(TEST_PROMPTS)} prompts"
    )
    print(f"🤖 Models: {MODELS}")
    print(f"🌡️ Temperature: {TEMPERATURE}")
    print(f"🎯 Optimal parameters: {OPTIMAL_PARAMS}")

    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_SCRIPT)],
        env=os.environ.copy(),
    )

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: {scenario: [] for scenario in TEST_PROMPTS}
        for model in MODELS
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

                for model in MODELS:
                    print(f"  🤖 Model: {model}")
                    for scenario in SCENARIO_ORDER:
                        prompt = TEST_PROMPTS[scenario]
                        print(f"    > Strat: {scenario:<15}...", end="", flush=True)

                        record: dict[str, Any] = {
                            "rep": rep,
                            "seed": rep,
                            "model": model,
                            "temperature": TEMPERATURE,
                            "prompt_category": scenario,
                            **optimal_outputs[scenario],
                        }

                        if record.get("optimal_objective") is None:
                            record["error"] = record.get("error", "Could not parse optimal objective")
                            results_db[model][scenario].append(record)
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
                                model=model,
                                messages=messages,
                                tools=ollama_tools,
                                options={"seed": rep, "temperature": TEMPERATURE},
                            )

                            tool_calls = response["message"].get("tool_calls", [])
                            if not tool_calls:
                                record["error"] = "No tool call from agent"
                                results_db[model][scenario].append(record)
                                print(" ❌ Agent failed (No Tool)")
                                continue

                            tool_call = tool_calls[0]
                            agent_fn_name = tool_call["function"]["name"]
                            agent_fn_args = tool_call["function"]["arguments"]
                            record["agent_tool_args"] = agent_fn_args

                            if agent_fn_name != "solve_allocation":
                                record["error"] = f"Agent called unexpected tool: {agent_fn_name}"
                                results_db[model][scenario].append(record)
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
                                results_db[model][scenario].append(record)
                                print(" ⚠️ Missing objective")
                                continue

                            record.update(compute_regret(agent_objective, record["optimal_objective"]))
                            results_db[model][scenario].append(record)
                            print(" ✅")

                        except Exception as exc:
                            record["error"] = f"Agent run error: {exc}"
                            results_db[model][scenario].append(record)
                            print(f" ❌ Agent error: {exc}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_average_regret_table(results_db)
    write_regret_table_csv(table_rows)
    write_regret_table_markdown(table_rows)
    print_regret_table(table_rows)

    print(f"\n✅ Regret model sweep complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Average regret table CSV saved to {REGRET_TABLE_CSV}")
    print(f"✅ Average regret table Markdown saved to {REGRET_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_regret_model_sweep())
