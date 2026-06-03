#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for parameter-selection accuracy paper experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

TOLERANCE = 0.1

TEST_PROMPTS = {
    "focus_cost": "Optimise this using all datacentres. My absolute priority is minimising cost, regardless of QoE or energy usage.",
    "focus_energy": "Optimise this using all datacentres. Minimise Energy usage above all else.",
    "focus_latency": "Optimise this using all datacentres. We need the best possible user experience (QoE)",
    "balanced": "Optimise this using all datacentres and balanced approach between cost, energy and latency.",
}

SCENARIO_LABELS = {
    "focus_cost": "Cost",
    "focus_energy": "Energy",
    "focus_latency": "QoE",
    "balanced": "Balanced",
}
SCENARIO_ORDER = ["focus_cost", "focus_energy", "focus_latency", "balanced"]

OPTIMAL_PARAMS = {
    "focus_cost": {"alpha": 0.0, "beta": 1.0, "max_datacentres": 9999},
    "focus_energy": {"alpha": 0.0, "beta": 0.0, "max_datacentres": 9999},
    "focus_latency": {"alpha": 1.0, "beta": 0.5, "max_datacentres": 9999},
    "balanced": {"alpha": 0.1, "beta": 0.5, "max_datacentres": 9999},
}

SYSTEM_PROMPT = """
You are the Intelligent Operations Manager for FlexMedia, a distributed split-rendering service.
Your goal is to balance Infrastructure Costs (opening Data Centres) and Delivery Costs,
against Energy Usage and User Quality of Experience (Latency).

YOU HAVE ACCESS TO THESE TOOLS:
1. solve_allocation(alpha, beta, max_datacentres = 9999): Run the optimization engine.
   Example call: solve_allocation(alpha = 1.0, beta = 1.0, max_datacentres = 9999)

GUIDELINES FOR PARAMETERS:
ALPHA (QoE TRADEOFF)
- alpha = 0.0 "Budget Mode". Prioritizes low cost. High latency risk.
- alpha = 0.1: "Balanced Mode". Good compromise.
- alpha >= 1.0: "Performance Mode". Prioritizes low latency (QoE). High cost.
- Do not set alpha above 100.0 under any circumstances.
BETA (Cost vs Energy): 0.0 to 1.0.
- beta = 1.0: "Budget Mode". Minimise Financial Cost.
- beta = 0.0: "Green Mode". Minimise Energy Usage.
- beta = 0.5: Balanced. Also use this when minimising QoE.

BEHAVIOUR:
- If the user gives a constraint in natural language interpret it in terms of parameters.
- If the user asks to only optimise for cost then set alpha = 0.0, beta = 1.0.
- If the user asks to focus on the environment or to be green, set alpha = 0.0, beta = 0.0.
- If the user asks to focus on QoE, set alpha = 1.0, beta = 0.5.
- If the user asks to use a balanced approach, set alpha = 0.1, beta = 0.5.
- If the user does not set max_datacentres then use 9999.

DO / DO NOT RULES:
- Do not set alpha above 100.0.
- Do set max_datacentres to be an integer; if unsure use 9999.
- Do not set max_datacentres to be None or Null.
"""

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
                        "description": "Weight for Latency/QoE. 0.0=Ignore Latency, 100.0=Critical.",
                    },
                    "beta": {
                        "type": "number",
                        "description": "Tradeoff between Financial Cost (1.0) and Energy (0.0). 0.5 is balanced.",
                    },
                    "max_datacentres": {
                        "type": "integer",
                        "description": "Max number of DCs to open. Use 9999 when unconstrained.",
                    },
                },
                "required": ["alpha", "beta", "max_datacentres"],
            },
        },
    }
]


def average(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None when no values are available."""
    if not values:
        return None
    return sum(values) / len(values)


def is_within_tolerance(actual: Any, target: float, tolerance: float = TOLERANCE) -> bool:
    """Return True when an actual parameter value is within the tolerance band."""
    if actual is None:
        return False
    try:
        return math.isclose(float(actual), target, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def score_parameter_selection(
    scenario: str,
    agent_args: dict[str, Any] | None,
    tolerance: float = TOLERANCE,
) -> dict[str, bool | float]:
    """Score alpha and beta jointly as 1.0, 0.5, or 0.0 for a scenario."""
    target = OPTIMAL_PARAMS[scenario]
    agent_args = agent_args or {}
    alpha_correct = is_within_tolerance(agent_args.get("alpha"), target["alpha"], tolerance)
    beta_correct = is_within_tolerance(agent_args.get("beta"), target["beta"], tolerance)

    return {
        "alpha_correct": alpha_correct,
        "beta_correct": beta_correct,
        "parameter_selection_score": (float(alpha_correct) + float(beta_correct)) / 2.0,
    }


def build_accuracy_table(results: dict[str, Any], columns: list[str]) -> list[dict[str, str]]:
    """Create scenario rows and sweep-value columns with mean joint parameter accuracy."""
    table_rows: list[dict[str, str]] = []

    for scenario in SCENARIO_ORDER:
        row = {"scenario": SCENARIO_LABELS[scenario]}
        for column in columns:
            runs = results.get(column, {}).get(scenario, [])
            scores = [
                float(run["parameter_selection_score"])
                for run in runs
                if run.get("parameter_selection_score") is not None
            ]
            accuracy = average(scores)
            row[column] = "" if accuracy is None else f"{accuracy:.4f}"
        table_rows.append(row)

    return table_rows


def write_accuracy_table_csv(table_rows: list[dict[str, str]], columns: list[str], path: Path) -> None:
    """Write a parameter-selection accuracy table as CSV."""
    fieldnames = ["scenario", *columns]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_accuracy_table_markdown(table_rows: list[dict[str, str]], columns: list[str], path: Path) -> None:
    """Write a parameter-selection accuracy table as Markdown."""
    headers = ["Scenario", *columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in table_rows:
        values = [row["scenario"], *[row[column] for column in columns]]
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_accuracy_table(table_rows: list[dict[str, str]], columns: list[str]) -> None:
    """Print a parameter-selection accuracy table to stdout."""
    headers = ["Scenario", *columns]
    rows = [[row["scenario"], *[row[column] for column in columns]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print("\nAverage parameter-selection accuracy (1.0 = alpha and beta correct):")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))
