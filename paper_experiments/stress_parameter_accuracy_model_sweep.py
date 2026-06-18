#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stress-test parameter-selection accuracy for cost-latency and cost-energy prompts.

This experiment is intentionally parameter-only: it asks each configured local
Ollama model to choose ``solve_allocation`` arguments for generated stress-prompt
variants, scores the selected alpha/beta values against the expected stress
settings, and does not run Gurobi or the allocation solver.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_selection_accuracy import (  # noqa: E402
    SYSTEM_PROMPT,
    TOLERANCE,
    TOOL_DEFINITION,
    average,
    is_within_tolerance,
)

# --- CONFIGURATION ---
N_REPS = 250
TEMPERATURE = 0.2
PROMPT_FUZZ_MODEL = "llama3.2:latest"
PROMPT_FUZZ_TEMPERATURE = 0.8
MODELS = [
    "interstellarninja/llama3.1-8b-tools:latest",
    "llama3.2:1b",
    "llama3.2:latest",
    "aratan/qwen3-4b-tools:latest",
    "lukaspetrik/gemma3-tools:4b",
]
MAX_DATACENTRES = 9999

STRESS_PROMPTS = {
    "stress_cost_latency": (
        "Optimise this using all datacentres. Stress-test the balancing behaviour: "
        "minimise financial cost while also minimising latency for users."
    ),
    "stress_cost_energy": (
        "Optimise this using all datacentres. Stress-test the balancing behaviour: "
        "minimise financial cost while also minimising energy consumption."
    ),
}

STRESS_SCENARIO_LABELS = {
    "stress_cost_latency": "Stress Cost-Latency",
    "stress_cost_energy": "Stress Cost-Energy",
}
STRESS_SCENARIO_ORDER = [
    "stress_cost_latency",
    "stress_cost_energy",
]

STRESS_OPTIMAL_PARAMS = {
    "stress_cost_latency": {
        "alpha": 0.5,
        "beta": 0.0,
        "max_datacentres": MAX_DATACENTRES,
    },
    "stress_cost_energy": {
        "alpha": 0.0,
        "beta": 0.5,
        "max_datacentres": MAX_DATACENTRES,
    },
}

STRESS_SYSTEM_PROMPT = SYSTEM_PROMPT + """

STRESS-TEST BALANCING RULES:
- When the user asks to minimise both financial cost and latency, perform a balanced
  stress-test act by setting alpha = 0.5 and beta = 0.0.
- When the user asks to minimise both financial cost and energy usage, perform a
  balanced stress-test act by setting alpha = 0.0 and beta = 0.5.
- These stress-test rules take precedence over the canonical scenario rules above
  whenever the user explicitly combines those objectives.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = (
    EXPERIMENT_DATA_DIR / "stress_parameter_accuracy_model_sweep_results.json"
)
ACCURACY_TABLE_CSV = (
    EXPERIMENT_DATA_DIR / "stress_parameter_accuracy_model_sweep_accuracy_table.csv"
)
ACCURACY_TABLE_MD = (
    EXPERIMENT_DATA_DIR / "stress_parameter_accuracy_model_sweep_accuracy_table.md"
)

STRESS_TOOL_DEFINITION = copy.deepcopy(TOOL_DEFINITION)
STRESS_TOOL_DEFINITION[0]["function"]["description"] += (
    " Stress-test presets: when asked to minimise cost and latency together, "
    "use alpha=0.5 beta=0.0; when asked to minimise cost and energy together, "
    "use alpha=0.0 beta=0.5."
)


def clean_generated_prompt(content: str) -> str:
    """Normalise a model-generated prompt to a single prompt string."""
    cleaned = content.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()

    cleaned = cleaned.strip(" \t\n\r\"'")
    return " ".join(cleaned.split())


def stress_prompt_generation_seed(rep: int, scenario: str) -> int:
    """Return a deterministic Ollama seed for a generated stress prompt variant."""
    return (rep * 1000) + STRESS_SCENARIO_ORDER.index(scenario)


def generate_stress_prompt_variant(
    scenario: str, canonical_prompt: str, rep: int
) -> dict[str, Any]:
    """Rewrite a stress-test prompt while preserving its two-objective intent.

    The generated wording varies each repetition without revealing the target
    alpha/beta values. If prompt generation fails, the experiment falls back to
    the canonical prompt and records the generation error.
    """
    import ollama

    seed = stress_prompt_generation_seed(rep, scenario)
    record: dict[str, Any] = {
        "original_prompt": canonical_prompt,
        "prompt_generation_model": PROMPT_FUZZ_MODEL,
        "prompt_generation_temperature": PROMPT_FUZZ_TEMPERATURE,
        "prompt_generation_seed": seed,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite stress-test prompts for an optimisation experiment. "
                "Preserve the exact same two optimisation objectives, the stress-test framing, "
                "and the instruction to use all datacentres. Do not introduce new goals, "
                "remove either objective, mention alpha/beta values, or solve the problem. "
                "Return only one rewritten user prompt with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scenario key: {scenario}\n"
                f"Canonical prompt: {canonical_prompt}\n\n"
                "Rewrite the canonical stress prompt in different wording while asking for the same optimisation."
            ),
        },
    ]

    try:
        response = ollama.chat(
            model=PROMPT_FUZZ_MODEL,
            messages=messages,
            options={"seed": seed, "temperature": PROMPT_FUZZ_TEMPERATURE},
        )
        generated_prompt = clean_generated_prompt(
            response["message"].get("content", "")
        )
        if not generated_prompt:
            raise ValueError("Prompt fuzzing model returned empty content")

        record["prompt"] = generated_prompt
        return record
    except Exception as exc:
        record["prompt"] = canonical_prompt
        record["prompt_generation_error"] = str(exc)
        return record


def score_stress_parameter_selection(
    scenario: str,
    agent_args: dict[str, Any] | None,
    tolerance: float = TOLERANCE,
) -> dict[str, bool | float]:
    """Score selected alpha/beta against the stress-test target for a scenario."""
    target = STRESS_OPTIMAL_PARAMS[scenario]
    agent_args = agent_args or {}
    alpha_correct = is_within_tolerance(
        agent_args.get("alpha"), target["alpha"], tolerance
    )
    beta_correct = is_within_tolerance(
        agent_args.get("beta"), target["beta"], tolerance
    )

    return {
        "alpha_correct": alpha_correct,
        "beta_correct": beta_correct,
        "parameter_selection_score": (float(alpha_correct) + float(beta_correct)) / 2.0,
    }


def build_stress_accuracy_table(results: dict[str, Any]) -> list[dict[str, str]]:
    """Create stress-scenario rows and model columns with mean joint accuracy."""
    table_rows: list[dict[str, str]] = []

    for scenario in STRESS_SCENARIO_ORDER:
        row = {"scenario": STRESS_SCENARIO_LABELS[scenario]}
        for model in MODELS:
            runs = results.get(model, {}).get(scenario, [])
            scores = [
                float(run["parameter_selection_score"])
                for run in runs
                if run.get("parameter_selection_score") is not None
            ]
            accuracy = average(scores)
            row[model] = "" if accuracy is None else f"{accuracy:.4f}"
        table_rows.append(row)

    return table_rows


def write_stress_accuracy_table_csv(table_rows: list[dict[str, str]]) -> None:
    """Write the stress-test parameter-selection accuracy table as CSV."""
    fieldnames = ["scenario", *MODELS]
    with ACCURACY_TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_stress_accuracy_table_markdown(table_rows: list[dict[str, str]]) -> None:
    """Write the stress-test parameter-selection accuracy table as Markdown."""
    headers = ["Scenario", *MODELS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in table_rows:
        values = [row["scenario"], *[row[model] for model in MODELS]]
        lines.append("| " + " | ".join(values) + " |")

    ACCURACY_TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_stress_accuracy_table(table_rows: list[dict[str, str]]) -> None:
    """Print a stress-test parameter-selection accuracy table to stdout."""
    headers = ["Scenario", *MODELS]
    rows = [[row["scenario"], *[row[model] for model in MODELS]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    print(
        "\nAverage stress-test parameter-selection accuracy (1.0 = alpha and beta correct):"
    )
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


async def run_stress_parameter_accuracy_model_sweep() -> None:
    """Run parameter-only stress testing for generated prompts and configured Ollama models."""
    import ollama

    print(
        f" Stress parameter accuracy model sweep: {N_REPS} reps x "
        f"{len(MODELS)} models x {len(STRESS_PROMPTS)} prompts"
    )
    print(f" Models: {MODELS}")
    print(f" Agent temperature: {TEMPERATURE}")
    print(
        f" Prompt fuzzing model: {PROMPT_FUZZ_MODEL} temp={PROMPT_FUZZ_TEMPERATURE}"
    )
    print(f" Stress-test target parameters: {STRESS_OPTIMAL_PARAMS}")
    print(f" Correct when alpha/beta are within ±{TOLERANCE} of target")
    print(" Parameter-selection only: no Gurobi or allocation solver calls are made")

    results_db: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: {scenario: [] for scenario in STRESS_PROMPTS} for model in MODELS
    }

    for rep in range(N_REPS):
        print(f"\n --- REP {rep + 1}/{N_REPS} ---")

        prompt_records: dict[str, dict[str, Any]] = {}
        for scenario in STRESS_SCENARIO_ORDER:
            canonical_prompt = STRESS_PROMPTS[scenario]
            print(f"   Stress prompt variant: {scenario:<20}...", end="", flush=True)
            prompt_record = generate_stress_prompt_variant(
                scenario, canonical_prompt, rep
            )
            prompt_records[scenario] = prompt_record
            print(
                " OK"
                if "prompt_generation_error" not in prompt_record
                else " WARNING fallback"
            )

        for model in MODELS:
            print(f"   Model: {model}")
            for scenario in STRESS_SCENARIO_ORDER:
                prompt_record = prompt_records[scenario]
                prompt = prompt_record["prompt"]
                print(f"    > Stress: {scenario:<20}...", end="", flush=True)

                record: dict[str, Any] = {
                    "rep": rep,
                    "seed": rep,
                    "model": model,
                    "temperature": TEMPERATURE,
                    "prompt_category": scenario,
                    "optimal_tool_args": STRESS_OPTIMAL_PARAMS[scenario],
                    "tolerance": TOLERANCE,
                    **prompt_record,
                }

                messages = [
                    {"role": "system", "content": STRESS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                try:
                    response = ollama.chat(
                        model=model,
                        messages=messages,
                        tools=STRESS_TOOL_DEFINITION,
                        options={"seed": rep, "temperature": TEMPERATURE},
                    )

                    tool_calls = response["message"].get("tool_calls", [])
                    if not tool_calls:
                        record["error"] = "No tool call from agent"
                        record.update(score_stress_parameter_selection(scenario, None))
                        results_db[model][scenario].append(record)
                        print(" ERROR Agent failed (No Tool)")
                        continue

                    tool_call = tool_calls[0]
                    agent_fn_name = tool_call["function"]["name"]
                    agent_fn_args = tool_call["function"]["arguments"]
                    record["agent_tool_name"] = agent_fn_name
                    record["agent_tool_args"] = agent_fn_args

                    if agent_fn_name != "solve_allocation":
                        record["error"] = (
                            f"Agent called unexpected tool: {agent_fn_name}"
                        )
                        record.update(
                            score_stress_parameter_selection(scenario, agent_fn_args)
                        )
                        results_db[model][scenario].append(record)
                        print(f" WARNING Agent skipped ({agent_fn_name})")
                        continue

                    record.update(
                        score_stress_parameter_selection(scenario, agent_fn_args)
                    )
                    results_db[model][scenario].append(record)
                    print(f" OK score={record['parameter_selection_score']:.1f}")

                except Exception as exc:
                    record["error"] = f"Agent run error: {exc}"
                    record.update(score_stress_parameter_selection(scenario, None))
                    results_db[model][scenario].append(record)
                    print(f" ERROR Agent error: {exc}")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    table_rows = build_stress_accuracy_table(results_db)
    write_stress_accuracy_table_csv(table_rows)
    write_stress_accuracy_table_markdown(table_rows)
    print_stress_accuracy_table(table_rows)

    print(
        f"\nOK Stress parameter accuracy model sweep complete. Results saved to {RESULTS_FILE}"
    )
    print(f"OK Accuracy table CSV saved to {ACCURACY_TABLE_CSV}")
    print(f"OK Accuracy table Markdown saved to {ACCURACY_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_stress_parameter_accuracy_model_sweep())
