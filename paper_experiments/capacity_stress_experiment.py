#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark capacity-stress behaviour with a two-step agent workflow.

The experiment varies the viewer-to-compute-capacity ratio from 0.3 to 1.5 in
0.2 increments.  The agent always gets a first attempt to solve the optimisation.
For feasible ratios below 1.0, that first solve is scored by regret.  For ratios
at or above 1.0, the protected solver returns a fast infeasibility status; the
agent is then asked to make a second-step decision.  Calling
``flag_infeasible_status`` is a success, while retrying ``solve_allocation`` is a
failure because it ignored the infeasibility signal.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from parameter_selection_accuracy import OPTIMAL_PARAMS, TEST_PROMPTS
from regret_temperature_sweep import (
    MODEL,
    TEMPERATURES,
    call_solve_allocation,
    compute_regret,
    metric_regret_percent_for_summary,
)

N_REPS = 250
CAPACITY_RATIOS = [round(0.3 + 0.2 * i, 1) for i in range(7)]
INFEASIBLE_RATIO_THRESHOLD = 1.0
MODELS = [model.strip() for model in os.environ.get("CAPACITY_STRESS_MODELS", MODEL).split(",") if model.strip()]
SCENARIO = "focus_cost"
SCENARIO_PROMPT = TEST_PROMPTS[SCENARIO]
SCENARIO_OPTIMAL_PARAMS = OPTIMAL_PARAMS[SCENARIO]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "capacity_stress_results.json"
SUMMARY_TABLE_CSV = EXPERIMENT_DATA_DIR / "capacity_stress_model_capacity_table.csv"
SUMMARY_TABLE_MD = EXPERIMENT_DATA_DIR / "capacity_stress_model_capacity_table.md"
SERVER_SCRIPT = SCRIPT_DIR / "capacity_stress_mcp_server.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_scenario.py"
DATA_PATH = PROJECT_ROOT / "data" / "current_problem.json"

SOLVE_SYSTEM_PROMPT = """
You are the Intelligent Operations Manager for FlexMedia. Choose optimisation
parameters and call solve_allocation for the user's objective. This first step is
diagnostic: if the tool reports infeasible capacity, do not invent a solution.
"""

DECISION_SYSTEM_PROMPT = """
You are reviewing the result of a previous solve_allocation attempt. If the tool
result says STATUS: INFEASIBLE or says there is not enough usable datacentre
capacity for the viewers, call flag_infeasible_status and stop. If the previous
solve was successful, you may call solve_allocation again only if you genuinely
need to improve a feasible allocation.
"""

SOLVE_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "solve_allocation",
            "description": "Attempt the allocation and return either optimisation metrics or an infeasibility status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {"type": "number"},
                    "beta": {"type": "number"},
                    "max_datacentres": {"type": "integer"},
                },
                "required": ["alpha", "beta", "max_datacentres"],
            },
        },
    }
]

DECISION_TOOL_DEFINITION = [
    SOLVE_TOOL_DEFINITION[0],
    {
        "type": "function",
        "function": {
            "name": "flag_infeasible_status",
            "description": "Declare that the previous solve showed insufficient usable compute capacity.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def average(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def parse_infeasible_status(text: str) -> bool:
    """Return True when a tool response clearly indicates capacity infeasibility."""
    text_upper = text.upper()
    return "STATUS: INFEASIBLE" in text_upper or "INFEASIBLE" in text_upper


def set_capacity_ratio(ratio: float) -> dict[str, Any]:
    """Scale current_problem capacities so viewers / usable capacity equals ratio approximately."""
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    n_consumers = int(data["n_consumers"])
    n_datacentres = int(data["n_datacentres"])
    target_capacity = n_consumers / ratio
    capacity_per_dc = target_capacity / n_datacentres
    data["datacentre_capacities"] = [capacity_per_dc] * n_datacentres
    DATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    usable_capacity = int(sum(math.floor(value) for value in data["datacentre_capacities"]))
    achieved_ratio = n_consumers / usable_capacity if usable_capacity else math.inf
    return {
        "requested_capacity_ratio": ratio,
        "achieved_viewer_to_usable_capacity_ratio": achieved_ratio,
        "n_consumers": n_consumers,
        "usable_capacity": usable_capacity,
        "raw_capacity": float(sum(data["datacentre_capacities"])),
        "is_capacity_feasible": usable_capacity >= n_consumers,
    }


def build_user_prompt(base_prompt: str, capacity_meta: dict[str, Any]) -> str:
    return (
        f"{base_prompt}\n\n"
        f"Capacity stress context: viewers={capacity_meta['n_consumers']}, "
        f"usable_compute_capacity={capacity_meta['usable_capacity']}, "
        f"viewer_to_compute_capacity_ratio={capacity_meta['requested_capacity_ratio']:.1f}. "
        "First try solve_allocation. If that tool reports infeasible capacity, make a follow-up decision rather than looping."
    )


def call_agent(
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    rep: int,
    temperature: float,
) -> dict[str, Any]:
    import ollama

    return ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={"seed": rep, "temperature": temperature},
    )


async def first_solve_attempt(
    session: Any,
    model: str,
    rep: int,
    temperature: float,
    prompt: str,
) -> dict[str, Any]:
    """Ask the agent to make its mandatory initial solve attempt and execute it."""
    record: dict[str, Any] = {}
    response = call_agent(
        model,
        [{"role": "system", "content": SOLVE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        SOLVE_TOOL_DEFINITION,
        rep,
        temperature,
    )
    tool_calls = response["message"].get("tool_calls", [])
    if not tool_calls:
        record["first_step_error"] = "No solve_allocation tool call from agent"
        return record

    tool_call = tool_calls[0]
    tool_name = tool_call["function"]["name"]
    tool_args = tool_call["function"].get("arguments", {})
    record["first_step_tool_name"] = tool_name
    record["first_step_tool_args"] = tool_args
    if tool_name != "solve_allocation":
        record["first_step_error"] = f"Agent called unexpected first-step tool: {tool_name}"
        return record

    result = await session.call_tool("solve_allocation", tool_args)
    output_text = result.content[0].text
    metrics, objective = await call_solve_allocation_from_text(output_text)
    record.update(
        {
            "first_step_tool_output": output_text,
            "agent_metrics": metrics,
            "agent_objective": objective,
            "first_step_infeasible": parse_infeasible_status(output_text),
        }
    )
    return record


async def call_solve_allocation_from_text(output_text: str) -> tuple[dict[str, float], float | None]:
    from regret_temperature_sweep import parse_metrics_from_text, parse_objective_from_text

    return parse_metrics_from_text(output_text), parse_objective_from_text(output_text)


async def second_step_infeasibility_decision(
    model: str,
    rep: int,
    temperature: float,
    prompt: str,
    first_step_output: str,
) -> dict[str, Any]:
    """Ask the agent to react to an infeasible first solve result."""
    messages = [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": (
                "The previous solve_allocation attempt returned this tool result:\n"
                f"{first_step_output}\n\n"
                "Make the correct next tool call now."
            ),
        },
    ]
    response = call_agent(model, messages, DECISION_TOOL_DEFINITION, rep, temperature)
    tool_calls = response["message"].get("tool_calls", [])
    record: dict[str, Any] = {}
    if not tool_calls:
        record.update({"infeasibility_success": False, "decision_error": "No second-step tool call from agent"})
        return record

    tool_name = tool_calls[0]["function"]["name"]
    record["decision_tool_name"] = tool_name
    record["decision_tool_args"] = tool_calls[0]["function"].get("arguments", {})
    record["infeasibility_success"] = tool_name == "flag_infeasible_status"
    if tool_name == "solve_allocation":
        record["decision_error"] = "Agent retried solve_allocation after an infeasible first-step result"
    elif tool_name != "flag_infeasible_status":
        record["decision_error"] = f"Agent called unexpected second-step tool: {tool_name}"
    return record


def result_columns() -> list[str]:
    return MODELS


def summarise_results(results: dict[str, Any]) -> list[dict[str, str]]:
    """Build one model-by-capacity-ratio table for the cost-minimisation scenario.

    Feasible rows report average metric-wise regret percentage. Infeasible rows
    report the proportion of second-step decisions that correctly flagged
    infeasibility.
    """
    table_rows = []
    for ratio in CAPACITY_RATIOS:
        ratio_key = str(ratio)
        is_feasible_row = ratio < INFEASIBLE_RATIO_THRESHOLD
        row = {
            "capacity_ratio": ratio_key,
            "result_metric": (
                "avg_metric_regret_percent"
                if is_feasible_row
                else "infeasibility_detection_proportion"
            ),
        }
        for model in MODELS:
            runs = [
                run
                for temperature in TEMPERATURES
                for run in results[ratio_key][model][str(temperature)]
            ]
            if is_feasible_row:
                regrets = [
                    value
                    for run in runs
                    if (value := metric_regret_percent_for_summary(run)) is not None
                ]
                row[model] = "" if (avg := average(regrets)) is None else f"{avg:.4f}"
            else:
                outcomes = [bool(run.get("infeasibility_success")) for run in runs]
                row[model] = "" if not outcomes else f"{sum(outcomes) / len(outcomes):.4f}"
        table_rows.append(row)
    return table_rows


def write_summary_table(path_csv: Path, path_md: Path, rows: list[dict[str, str]]) -> None:
    columns = result_columns()
    fieldnames = ["capacity_ratio", "result_metric", *columns]
    with path_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"| Capacity ratio | Result metric | {' | '.join(columns)} |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([row["capacity_ratio"], row["result_metric"], *[row[column] for column in columns]])
            + " |"
        )
    path_md.write_text(
        "# Capacity-stress model comparison for cost minimisation\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


async def run_capacity_stress_experiment() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print(
        f"🚀 Capacity stress: {N_REPS} reps x {len(CAPACITY_RATIOS)} ratios x "
        f"{len(MODELS)} models x {len(TEMPERATURES)} temps x 1 scenario ({SCENARIO})"
    )
    server_params = StdioServerParameters(command="python", args=[str(SERVER_SCRIPT)], env=os.environ.copy())
    results: dict[str, Any] = {
        str(r): {
            model: {str(t): [] for t in TEMPERATURES}
            for model in MODELS
        }
        for r in CAPACITY_RATIOS
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for rep in range(N_REPS):
                for ratio in CAPACITY_RATIOS:
                    subprocess.run(["python", str(GENERATOR_SCRIPT), str(rep)], check=True, capture_output=True)
                    capacity_meta = set_capacity_ratio(ratio)
                    optimal_output = {}
                    if ratio < INFEASIBLE_RATIO_THRESHOLD:
                        metrics, objective = await call_solve_allocation(session, SCENARIO_OPTIMAL_PARAMS)
                        optimal_output = {
                            "optimal_metrics": metrics,
                            "optimal_objective": objective,
                            "optimal_tool_args": SCENARIO_OPTIMAL_PARAMS,
                        }
                    for model in MODELS:
                        for temperature in TEMPERATURES:
                            prompt = build_user_prompt(SCENARIO_PROMPT, capacity_meta)
                            record = {
                                "rep": rep,
                                "seed": rep,
                                "model": model,
                                "temperature": temperature,
                                "prompt_category": SCENARIO,
                                **capacity_meta,
                                **optimal_output,
                            }
                            first_step = await first_solve_attempt(session, model, rep, temperature, prompt)
                            record.update(first_step)

                            if ratio >= INFEASIBLE_RATIO_THRESHOLD:
                                decision = await second_step_infeasibility_decision(
                                    model,
                                    rep,
                                    temperature,
                                    prompt,
                                    first_step.get("first_step_tool_output", ""),
                                )
                                record.update(decision)
                            elif record.get("agent_objective") is not None:
                                record.update(
                                    compute_regret(
                                        record["agent_metrics"],
                                        record["optimal_metrics"],
                                        record["agent_objective"],
                                        record["optimal_objective"],
                                    )
                                )
                            else:
                                record["error"] = record.get("first_step_error", "Could not parse first-step objective")

                            results[str(ratio)][model][str(temperature)].append(record)

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    table_rows = summarise_results(results)
    write_summary_table(SUMMARY_TABLE_CSV, SUMMARY_TABLE_MD, table_rows)


if __name__ == "__main__":
    asyncio.run(run_capacity_stress_experiment())
