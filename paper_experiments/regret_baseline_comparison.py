#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare fixed-temperature agentic regret against non-agentic baselines.

For each generated world seed and prompt scenario, this script:
1) solves the scenario with the fixed optimal parameters for that scenario;
2) asks interstellarninja/llama3.1-8b-tools:latest at temperature 0.0 to choose solve_allocation parameters;
3) chooses parameters with deterministic keyword and semantic classifiers;
4) samples one random alpha/beta pair from the optimal-value grid;
5) computes objective regret percentage relative to the fixed optimum; and
6) writes both raw JSON results and average-regret tables.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from regret_temperature_sweep import (  # noqa: E402
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
MODEL = "interstellarninja/llama3.1-8b-tools:latest"
TEMPERATURE = 0.0
MAX_DATACENTRES = 9999

BASELINE_COLUMNS = [
    "Agentic interstellarninja/llama3.1-8b-tools:latest temp=0.0",
    "Keyword classifier",
    "Semantic classifier",
    "Random parameter selection",
]

PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_results.json"
REGRET_TABLE_CSV = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_average_regret_table.csv"
REGRET_TABLE_MD = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_average_regret_table.md"

SERVER_SCRIPT = PROJECT_ROOT / "server" / "mcp_server.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_scenario.py"

PARAMETER_GRID = {
    "alpha": sorted({params["alpha"] for params in OPTIMAL_PARAMS.values()}),
    "beta": sorted({params["beta"] for params in OPTIMAL_PARAMS.values()}),
    "max_datacentres": [MAX_DATACENTRES],
}

KEYWORD_RULES = [
    ("focus_cost", ("cost", "financial", "money", "price", "cheap", "cheapest", "budget")),
    ("focus_energy", ("energy", "green", "greener", "carbon", "emissions", "sustainable")),
    ("focus_latency", ("latency", "experience", "responsive", "speed", "fast", "performance")),
]

SEMANTIC_DIMENSION_TERMS = {
    "cost": {
        "cost": 2.0,
        "financial": 2.0,
        "budget": 1.5,
        "controlled": 1.0,
        "minimising": 0.5,
        "minimizing": 0.5,
        "priority": 0.5,
        "cheap": 1.5,
    },
    "energy": {
        "energy": 2.0,
        "usage": 0.5,
        "green": 2.0,
        "greener": 2.0,
        "carbon": 1.5,
        "sustainable": 1.5,
        "minimise": 0.5,
        "minimize": 0.5,
    },
    "latency": {
        "latency": 2.0,
        "experience": 2.0,
        "user": 1.0,
        "responsive": 1.5,
        "performance": 1.5,
        "fast": 1.5,
        "lowest": 1.0,
        "acceptable": 0.5,
        "poor": 0.5,
    },
}

SEMANTIC_DIMENSION_TO_SCENARIO = {
    "cost": "focus_cost",
    "energy": "focus_energy",
    "latency": "focus_latency",
}
SEMANTIC_TIE_BREAK_ORDER = ["cost", "energy", "latency"]


def build_average_regret_table(results: dict[str, Any]) -> list[dict[str, str]]:
    """Create rows with scenarios on rows and baselines on columns."""
    table_rows: list[dict[str, str]] = []

    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for baseline in BASELINE_COLUMNS:
            runs = results.get(baseline, {}).get(scenario, [])
            regrets = [
                float(run["objective_regret_percent"])
                for run in runs
                if run.get("objective_regret_percent") is not None
            ]
            avg_regret = average(regrets)
            row[baseline] = "" if avg_regret is None else f"{avg_regret:.4f}"
        table_rows.append(row)

    return table_rows


def write_regret_table_csv(table_rows: list[dict[str, str]]) -> None:
    """Write the average regret percentage table as CSV."""
    fieldnames = ["scenario", *BASELINE_COLUMNS]
    with REGRET_TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_regret_table_markdown(table_rows: list[dict[str, str]]) -> None:
    """Write the average regret percentage table as Markdown for quick viewing."""
    headers = ["Scenario", *BASELINE_COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in table_rows:
        values = [row["scenario"], *[row[baseline] for baseline in BASELINE_COLUMNS]]
        lines.append("| " + " | ".join(values) + " |")

    REGRET_TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_regret_table(table_rows: list[dict[str, str]]) -> None:
    """Print the average regret percentage table to stdout."""
    headers = ["Scenario", *BASELINE_COLUMNS]
    rows = [[row["scenario"], *[row[baseline] for baseline in BASELINE_COLUMNS]] for row in table_rows]
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


def keyword_classifier(prompt: str) -> dict[str, Any]:
    """Select the first single-objective parameter profile whose keywords occur in the prompt."""
    prompt_lower = prompt.lower()
    for scenario, keywords in KEYWORD_RULES:
        if any(keyword in prompt_lower for keyword in keywords):
            return dict(OPTIMAL_PARAMS[scenario])

    return dict(OPTIMAL_PARAMS["focus_cost"])


def semantic_classifier(prompt: str) -> dict[str, Any]:
    """Select a deterministic single-objective profile using lightweight semantic scores.

    The classifier intentionally resolves the prompt to one dominant objective dimension, so
    multi-objective tradeoff language is not represented as a combined parameter choice.
    """
    prompt_lower = prompt.lower()
    scores: dict[str, float] = {}

    for dimension, terms in SEMANTIC_DIMENSION_TERMS.items():
        scores[dimension] = sum(weight for term, weight in terms.items() if term in prompt_lower)

    best_dimension = max(
        SEMANTIC_TIE_BREAK_ORDER,
        key=lambda dimension: (scores[dimension], -SEMANTIC_TIE_BREAK_ORDER.index(dimension)),
    )
    return dict(OPTIMAL_PARAMS[SEMANTIC_DIMENSION_TO_SCENARIO[best_dimension]])


def random_parameter_selection(rep: int, scenario: str) -> dict[str, Any]:
    """Sample one alpha/beta pair uniformly from the possible optimal values."""
    rng = random.Random(f"{rep}:{scenario}")
    return {
        "alpha": rng.choice(PARAMETER_GRID["alpha"]),
        "beta": rng.choice(PARAMETER_GRID["beta"]),
        "max_datacentres": MAX_DATACENTRES,
    }


async def evaluate_parameter_baseline(
    session: Any,
    baseline_name: str,
    scenario: str,
    rep: int,
    optimal_output: dict[str, Any],
    parameter_selector: Callable[[], dict[str, Any]],
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Run solve_allocation for selected parameters and compute regret."""
    selected_params = parameter_selector()
    record: dict[str, Any] = {
        "rep": rep,
        "seed": rep,
        "baseline": baseline_name,
        "prompt_category": scenario,
        "selected_tool_args": selected_params,
        **optimal_output,
    }
    if sample_index is not None:
        record["sample_index"] = sample_index

    if record.get("optimal_objective") is None:
        record["error"] = record.get("error", "Could not parse optimal objective")
        return record

    try:
        selected_metrics, selected_objective = await call_solve_allocation(session, selected_params)
        record["selected_metrics"] = selected_metrics
        record["selected_objective"] = selected_objective

        if selected_objective is None:
            record["error"] = "Could not parse selected objective"
            return record

        record.update(compute_regret(selected_objective, record["optimal_objective"]))
        return record
    except Exception as exc:
        record["error"] = f"Parameter baseline error: {exc}"
        return record


async def evaluate_agentic_baseline(
    session: Any,
    ollama_tools: list[dict[str, Any]],
    scenario: str,
    prompt: str,
    rep: int,
    optimal_output: dict[str, Any],
) -> dict[str, Any]:
    """Ask the configured Ollama model for solve_allocation parameters and compute regret."""
    import ollama

    record: dict[str, Any] = {
        "rep": rep,
        "seed": rep,
        "baseline": BASELINE_COLUMNS[0],
        "model": MODEL,
        "temperature": TEMPERATURE,
        "prompt_category": scenario,
        **optimal_output,
    }

    if record.get("optimal_objective") is None:
        record["error"] = record.get("error", "Could not parse optimal objective")
        return record

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
            options={"seed": rep, "temperature": TEMPERATURE},
        )

        tool_calls = response["message"].get("tool_calls", [])
        if not tool_calls:
            record["error"] = "No tool call from agent"
            return record

        tool_call = tool_calls[0]
        agent_fn_name = tool_call["function"]["name"]
        agent_fn_args = tool_call["function"]["arguments"]
        record["agent_tool_args"] = agent_fn_args

        if agent_fn_name != "solve_allocation":
            record["error"] = f"Agent called unexpected tool: {agent_fn_name}"
            return record

        agent_metrics, agent_objective = await call_solve_allocation(session, agent_fn_args)
        record["agent_metrics"] = agent_metrics
        record["agent_objective"] = agent_objective

        if agent_objective is None:
            record["error"] = "Could not parse agent objective"
            return record

        record.update(compute_regret(agent_objective, record["optimal_objective"]))
        return record
    except Exception as exc:
        record["error"] = f"Agent run error: {exc}"
        return record


async def run_regret_baseline_comparison() -> None:
    """Run regret benchmarking for the agentic system and non-agentic baselines."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print(
        "🚀 Regret baseline comparison: "
        f"{N_REPS} reps x {len(TEST_PROMPTS)} prompts x "
        f"{len(BASELINE_COLUMNS)} baselines"
    )
    print(f"🤖 Agentic model: {MODEL}")
    print(f"🌡️ Agentic temperature: {TEMPERATURE}")
    print("🎲 Random samples per generated workload/scenario: 1")
    print(f"🎯 Optimal parameters: {OPTIMAL_PARAMS}")
    print(f"🎛️ Random parameter grid: {PARAMETER_GRID}")

    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_SCRIPT)],
        env=os.environ.copy(),
    )

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        baseline: {scenario: [] for scenario in TEST_PROMPTS}
        for baseline in BASELINE_COLUMNS
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

                for scenario in SCENARIO_ORDER:
                    prompt = TEST_PROMPTS[scenario]
                    optimal_output = optimal_outputs[scenario]
                    print(f"  > Strat: {scenario:<15}")

                    print(f"    - {BASELINE_COLUMNS[0]}...", end="", flush=True)
                    agent_record = await evaluate_agentic_baseline(
                        session,
                        ollama_tools,
                        scenario,
                        prompt,
                        rep,
                        optimal_output,
                    )
                    results_db[BASELINE_COLUMNS[0]][scenario].append(agent_record)
                    print(" ✅" if agent_record.get("objective_regret_percent") is not None else " ❌")

                    print(f"    - {BASELINE_COLUMNS[1]}...", end="", flush=True)
                    keyword_record = await evaluate_parameter_baseline(
                        session,
                        BASELINE_COLUMNS[1],
                        scenario,
                        rep,
                        optimal_output,
                        lambda prompt=prompt: keyword_classifier(prompt),
                    )
                    results_db[BASELINE_COLUMNS[1]][scenario].append(keyword_record)
                    print(" ✅" if keyword_record.get("objective_regret_percent") is not None else " ❌")

                    print(f"    - {BASELINE_COLUMNS[2]}...", end="", flush=True)
                    semantic_record = await evaluate_parameter_baseline(
                        session,
                        BASELINE_COLUMNS[2],
                        scenario,
                        rep,
                        optimal_output,
                        lambda prompt=prompt: semantic_classifier(prompt),
                    )
                    results_db[BASELINE_COLUMNS[2]][scenario].append(semantic_record)
                    print(" ✅" if semantic_record.get("objective_regret_percent") is not None else " ❌")

                    print(f"    - {BASELINE_COLUMNS[3]}...", end="", flush=True)
                    random_record = await evaluate_parameter_baseline(
                        session,
                        BASELINE_COLUMNS[3],
                        scenario,
                        rep,
                        optimal_output,
                        lambda rep=rep, scenario=scenario: random_parameter_selection(rep, scenario),
                    )
                    results_db[BASELINE_COLUMNS[3]][scenario].append(random_record)
                    print(" ✅" if random_record.get("objective_regret_percent") is not None else " ❌")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_average_regret_table(results_db)
    write_regret_table_csv(table_rows)
    write_regret_table_markdown(table_rows)
    print_regret_table(table_rows)

    print(f"\n✅ Regret baseline comparison complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Average regret table CSV saved to {REGRET_TABLE_CSV}")
    print(f"✅ Average regret table Markdown saved to {REGRET_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_regret_baseline_comparison())
