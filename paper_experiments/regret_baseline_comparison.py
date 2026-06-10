#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare fixed-temperature agentic regret against non-agentic baselines.

For each generated world seed and prompt scenario, this script:
1) solves the scenario with the fixed optimal parameters for that scenario;
2) rewrites the canonical scenario prompt with llama3.2:latest to fuzz the wording while
   preserving the same optimisation requirements;
3) asks aratan/qwen3-4b-tools:latest at temperature 0.2 to choose solve_allocation parameters;
4) chooses parameters with deterministic keyword and semantic classifiers from the fuzzed prompt;
5) samples one random alpha/beta pair from the optimal-value grid;
6) falls back to the keyword classifier when the agentic tool call fails, so the
   agentic baseline still yields a regret value for this experiment;
7) computes metric-wise absolute percentage regret for cost, energy, and latency relative to the fixed optimum; and
8) records whether each baseline selected the target alpha/beta parameters; and
9) writes raw JSON results plus average-regret and parameter-accuracy tables.
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

from parameter_selection_accuracy import (  # noqa: E402
    SYSTEM_PROMPT,
    TOOL_DEFINITION,
    build_accuracy_table,
    print_accuracy_table,
    score_parameter_selection,
    write_accuracy_table_csv,
    write_accuracy_table_markdown,
)
from regret_temperature_sweep import (  # noqa: E402
    OPTIMAL_PARAMS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    TEST_PROMPTS,
    average,
    call_solve_allocation,
    compute_regret,
    metric_regret_percent_for_summary,
)


# --- CONFIGURATION ---
N_REPS = 250
MODEL = "aratan/qwen3-4b-tools:latest"
TEMPERATURE = 0.2
PROMPT_FUZZ_MODEL = "llama3.2:latest"
PROMPT_FUZZ_TEMPERATURE = 0.8
MAX_DATACENTRES = 9999

BASELINE_COLUMNS = [
    f"Agentic {MODEL} temp={TEMPERATURE}",
    "Keyword classifier",
    "Semantic classifier",
    "Random parameter selection",
]

PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
RESULTS_FILE = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_results.json"
REGRET_TABLE_CSV = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_average_regret_table.csv"
REGRET_TABLE_MD = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_average_regret_table.md"
ACCURACY_TABLE_CSV = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_parameter_accuracy_table.csv"
ACCURACY_TABLE_MD = EXPERIMENT_DATA_DIR / "regret_baseline_comparison_parameter_accuracy_table.md"

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
                regret_percent
                for run in runs
                if (regret_percent := metric_regret_percent_for_summary(run)) is not None
            ]
            avg_regret = average(regrets)
            row[baseline] = "" if avg_regret is None else f"{avg_regret:.4f}"
        table_rows.append(row)

    return table_rows


def write_regret_table_csv(table_rows: list[dict[str, str]]) -> None:
    """Write the average absolute regret percentage table as CSV."""
    fieldnames = ["scenario", *BASELINE_COLUMNS]
    with REGRET_TABLE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def write_regret_table_markdown(table_rows: list[dict[str, str]]) -> None:
    """Write the average absolute regret percentage table as Markdown for quick viewing."""
    headers = ["Scenario", *BASELINE_COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in table_rows:
        values = [row["scenario"], *[row[baseline] for baseline in BASELINE_COLUMNS]]
        lines.append("| " + " | ".join(values) + " |")

    REGRET_TABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_regret_table(table_rows: list[dict[str, str]]) -> None:
    """Print the average absolute regret percentage table to stdout."""
    headers = ["Scenario", *BASELINE_COLUMNS]
    rows = [[row["scenario"], *[row[baseline] for baseline in BASELINE_COLUMNS]] for row in table_rows]
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print("\nAverage metric-wise absolute regret percentage from optimum:")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def score_record_parameter_selection(
    record: dict[str, Any],
    scenario: str,
    selected_args: dict[str, Any] | None,
) -> None:
    """Store alpha/beta correctness for the parameters actually selected by a baseline.

    Agentic failures are scored as incorrect rather than scoring the keyword fallback that
    is used solely to keep the regret comparison table complete.
    """
    record.update(score_parameter_selection(scenario, selected_args))


def prompt_generation_seed(rep: int, scenario: str) -> int:
    """Return a deterministic Ollama seed for a generated prompt variant."""
    return (rep * 1000) + SCENARIO_ORDER.index(scenario)


def clean_generated_prompt(content: str) -> str:
    """Normalise a model-generated prompt to a single prompt string."""
    cleaned = content.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()

    cleaned = cleaned.strip(" \t\n\r\"\'")
    return " ".join(cleaned.split())


def generate_fuzzed_prompt(scenario: str, canonical_prompt: str, rep: int) -> dict[str, Any]:
    """Rewrite a canonical scenario prompt while preserving its optimisation intent.

    The fuzzing model is deliberately separated from the agentic tool-calling model so
    the baseline comparison evaluates prompt robustness rather than a single fixed
    wording of each scenario. If prompt generation fails, the experiment falls back to
    the canonical prompt and records the generation error in the result JSON.
    """
    import ollama

    seed = prompt_generation_seed(rep, scenario)
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
                "You rewrite benchmark prompts for an optimisation experiment. "
                "Preserve the exact same requirements, objective priorities, trade-offs, "
                "and the instruction to use all datacentres. Do not introduce new goals, "
                "remove existing goals, mention alpha/beta values, or solve the problem. "
                "Return only one rewritten user prompt with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scenario key: {scenario}\n"
                f"Canonical prompt: {canonical_prompt}\n\n"
                "Rewrite the canonical prompt in different wording while asking for the same optimisation."
            ),
        },
    ]

    try:
        response = ollama.chat(
            model=PROMPT_FUZZ_MODEL,
            messages=messages,
            options={"seed": seed, "temperature": PROMPT_FUZZ_TEMPERATURE},
        )
        generated_prompt = clean_generated_prompt(response["message"].get("content", ""))
        if not generated_prompt:
            raise ValueError("Prompt fuzzing model returned empty content")

        record["prompt"] = generated_prompt
        return record
    except Exception as exc:
        record["prompt"] = canonical_prompt
        record["prompt_generation_error"] = str(exc)
        return record


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
    prompt_record: dict[str, Any],
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
        **prompt_record,
        **optimal_output,
    }
    if sample_index is not None:
        record["sample_index"] = sample_index

    score_record_parameter_selection(record, scenario, selected_params)

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

        record.update(
            compute_regret(
                selected_metrics,
                record["optimal_metrics"],
                selected_objective,
                record["optimal_objective"],
            )
        )
        return record
    except Exception as exc:
        record["error"] = f"Parameter baseline error: {exc}"
        return record


async def apply_agent_keyword_fallback(
    session: Any,
    record: dict[str, Any],
    prompt: str,
    reason: str,
) -> dict[str, Any]:
    """Use the keyword classifier when the agentic tool call fails in this experiment.

    This fallback is intentionally scoped to the regret baseline comparison because the
    experiment needs a regret value for every generated prompt. The raw record keeps the
    original agent failure reason and marks that the reported regret came from the
    keyword fallback rather than a successful agent tool call.
    """
    record["agent_tool_call_failed"] = True
    record["agent_tool_call_failure_reason"] = reason
    record["fallback_used"] = True
    record["fallback_baseline"] = "Keyword classifier"

    if record.get("parameter_selection_score") is None:
        score_record_parameter_selection(record, record["prompt_category"], None)

    fallback_params = keyword_classifier(prompt)
    record["fallback_tool_args"] = fallback_params

    try:
        fallback_metrics, fallback_objective = await call_solve_allocation(session, fallback_params)
        record["fallback_metrics"] = fallback_metrics
        record["fallback_objective"] = fallback_objective

        if fallback_objective is None:
            record["error"] = "Keyword fallback could not parse selected objective"
            return record

        record.update(
            compute_regret(
                fallback_metrics,
                record["optimal_metrics"],
                fallback_objective,
                record["optimal_objective"],
            )
        )
        return record
    except Exception as exc:
        record["error"] = f"Keyword fallback error after agent failure ({reason}): {exc}"
        return record


async def evaluate_agentic_baseline(
    session: Any,
    scenario: str,
    prompt: str,
    rep: int,
    optimal_output: dict[str, Any],
    prompt_record: dict[str, Any],
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
        **prompt_record,
        **optimal_output,
    }

    if record.get("optimal_objective") is None:
        score_record_parameter_selection(record, scenario, None)
        record["error"] = record.get("error", "Could not parse optimal objective")
        return record

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITION,
            options={"seed": rep, "temperature": TEMPERATURE},
        )

        tool_calls = response["message"].get("tool_calls", [])
        if not tool_calls:
            return await apply_agent_keyword_fallback(session, record, prompt, "No tool call from agent")

        tool_call = tool_calls[0]
        agent_fn_name = tool_call["function"]["name"]
        agent_fn_args = tool_call["function"]["arguments"]
        record["agent_tool_args"] = agent_fn_args
        score_record_parameter_selection(record, scenario, agent_fn_args)

        if agent_fn_name != "solve_allocation":
            return await apply_agent_keyword_fallback(
                session,
                record,
                prompt,
                f"Agent called unexpected tool: {agent_fn_name}",
            )

        agent_metrics, agent_objective = await call_solve_allocation(session, agent_fn_args)
        record["agent_metrics"] = agent_metrics
        record["agent_objective"] = agent_objective

        if agent_objective is None:
            return await apply_agent_keyword_fallback(session, record, prompt, "Could not parse agent objective")

        record.update(
            compute_regret(
                agent_metrics,
                record["optimal_metrics"],
                agent_objective,
                record["optimal_objective"],
            )
        )
        return record
    except Exception as exc:
        return await apply_agent_keyword_fallback(session, record, prompt, f"Agent run error: {exc}")


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
    print(f"🧪 Prompt fuzzing model: {PROMPT_FUZZ_MODEL} temp={PROMPT_FUZZ_TEMPERATURE}")
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
                    canonical_prompt = TEST_PROMPTS[scenario]
                    optimal_output = optimal_outputs[scenario]
                    print(f"  > Strat: {scenario:<15}")

                    print(f"    - prompt fuzz via {PROMPT_FUZZ_MODEL}...", end="", flush=True)
                    prompt_record = generate_fuzzed_prompt(scenario, canonical_prompt, rep)
                    prompt = prompt_record["prompt"]
                    print(" ✅" if "prompt_generation_error" not in prompt_record else " ⚠️ fallback")

                    print(f"    - {BASELINE_COLUMNS[0]}...", end="", flush=True)
                    agent_record = await evaluate_agentic_baseline(
                        session,
                        scenario,
                        prompt,
                        rep,
                        optimal_output,
                        prompt_record,
                    )
                    results_db[BASELINE_COLUMNS[0]][scenario].append(agent_record)
                    if agent_record.get("metric_regret_percent") is None:
                        print(" ❌")
                    elif agent_record.get("fallback_used"):
                        print(" ⚠️ keyword fallback")
                    else:
                        print(" ✅")

                    print(f"    - {BASELINE_COLUMNS[1]}...", end="", flush=True)
                    keyword_record = await evaluate_parameter_baseline(
                        session,
                        BASELINE_COLUMNS[1],
                        scenario,
                        rep,
                        optimal_output,
                        lambda prompt=prompt: keyword_classifier(prompt),
                        prompt_record,
                    )
                    results_db[BASELINE_COLUMNS[1]][scenario].append(keyword_record)
                    print(" ✅" if keyword_record.get("metric_regret_percent") is not None else " ❌")

                    print(f"    - {BASELINE_COLUMNS[2]}...", end="", flush=True)
                    semantic_record = await evaluate_parameter_baseline(
                        session,
                        BASELINE_COLUMNS[2],
                        scenario,
                        rep,
                        optimal_output,
                        lambda prompt=prompt: semantic_classifier(prompt),
                        prompt_record,
                    )
                    results_db[BASELINE_COLUMNS[2]][scenario].append(semantic_record)
                    print(" ✅" if semantic_record.get("metric_regret_percent") is not None else " ❌")

                    print(f"    - {BASELINE_COLUMNS[3]}...", end="", flush=True)
                    random_record = await evaluate_parameter_baseline(
                        session,
                        BASELINE_COLUMNS[3],
                        scenario,
                        rep,
                        optimal_output,
                        lambda rep=rep, scenario=scenario: random_parameter_selection(rep, scenario),
                        prompt_record,
                    )
                    results_db[BASELINE_COLUMNS[3]][scenario].append(random_record)
                    print(" ✅" if random_record.get("metric_regret_percent") is not None else " ❌")

    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results_db, f, indent=2)

    regret_table_rows = build_average_regret_table(results_db)
    write_regret_table_csv(regret_table_rows)
    write_regret_table_markdown(regret_table_rows)
    print_regret_table(regret_table_rows)

    accuracy_table_rows = build_accuracy_table(results_db, BASELINE_COLUMNS)
    write_accuracy_table_csv(accuracy_table_rows, BASELINE_COLUMNS, ACCURACY_TABLE_CSV)
    write_accuracy_table_markdown(accuracy_table_rows, BASELINE_COLUMNS, ACCURACY_TABLE_MD)
    print_accuracy_table(accuracy_table_rows, BASELINE_COLUMNS)

    print(f"\n✅ Regret baseline comparison complete. Results saved to {RESULTS_FILE}")
    print(f"✅ Average absolute regret table CSV saved to {REGRET_TABLE_CSV}")
    print(f"✅ Average absolute regret table Markdown saved to {REGRET_TABLE_MD}")
    print(f"✅ Parameter-selection accuracy table CSV saved to {ACCURACY_TABLE_CSV}")
    print(f"✅ Parameter-selection accuracy table Markdown saved to {ACCURACY_TABLE_MD}")


if __name__ == "__main__":
    asyncio.run(run_regret_baseline_comparison())
