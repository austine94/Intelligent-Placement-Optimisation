#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure and plot semantic drift in varying-prompt regret experiments.

The varying-prompt regret sweep stores each generated prompt variant in the raw
results JSON alongside the canonical ``original_prompt``. This post-processing
script deduplicates those prompt records by scenario/repetition, computes the
cosine distance between each original prompt and its generated prompt, and writes
per-scenario summaries plus bar plots.

By default the script tries to use ``sentence-transformers`` embeddings for a
semantic sentence-level distance. If that optional dependency or its requested
model is unavailable, it falls back to a deterministic TF-IDF word/character
n-gram cosine distance implemented with the Python standard library. Use
``--backend sentence-transformers`` to require embedding-based distances.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_semantic_distance import (  # noqa: E402
    Backend,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    pair_distances,
)
from regret_temperature_sweep import SCENARIO_LABELS, SCENARIO_ORDER, TEST_PROMPTS, average  # noqa: E402
from varying_prompt_regret_model_sweep import RESULTS_FILE  # noqa: E402

EXPERIMENT_DATA_DIR = SCRIPT_DIR / "experiment_data"
PLOTS_DIR = SCRIPT_DIR / "paper_plots"
SUMMARY_CSV = EXPERIMENT_DATA_DIR / "varying_prompt_semantic_distance_summary.csv"
SUMMARY_MD = EXPERIMENT_DATA_DIR / "varying_prompt_semantic_distance_summary.md"
DETAILS_CSV = EXPERIMENT_DATA_DIR / "varying_prompt_semantic_distance_details.csv"
COMBINED_PLOT = PLOTS_DIR / "varying_prompt_semantic_distance_by_scenario.pdf"
INDIVIDUAL_PLOT_DIR = PLOTS_DIR / "varying_prompt_semantic_distance_by_scenario"

SCENARIO_COLOURS: list[tuple[float, float, float]] = [
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
]


@dataclass(frozen=True)
class PromptPair:
    """A canonical/generated prompt pair from one scenario repetition."""

    scenario: str
    rep: int | None
    prompt_generation_seed: int | None
    original_prompt: str
    generated_prompt: str
    prompt_generation_error: str | None = None


@dataclass(frozen=True)
class DistanceRecord:
    """A prompt pair plus its computed cosine distance."""

    scenario: str
    rep: int | None
    prompt_generation_seed: int | None
    original_prompt: str
    generated_prompt: str
    distance: float
    prompt_generation_error: str | None = None


class PdfPage:
    """Tiny PDF writer with enough drawing primitives for bar charts."""

    def __init__(self, width: float = 760, height: float = 504):
        self.width = width
        self.height = height
        self.commands: list[str] = []

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text(self, x: float, y: float, text: str, size: int = 10) -> None:
        self.commands.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({self._escape(text)}) Tj ET")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float = 1.0,
        colour: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        r, g, b = colour
        self.commands.append(
            f"{r:.2f} {g:.2f} {b:.2f} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        )

    def fill_rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        colour: tuple[float, float, float],
    ) -> None:
        r, g, b = colour
        self.commands.append(
            f"{r:.2f} {g:.2f} {b:.2f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f"
        )

    def save(self, output_file: Path) -> None:
        """Write the page to a single-page PDF file."""
        content = "\n".join(self.commands).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ).encode("latin-1"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(content)).encode("latin-1") + b" >>\nstream\n" + content + b"\nendstream",
        ]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("wb") as f:
            f.write(b"%PDF-1.4\n")
            offsets = [0]
            for index, obj in enumerate(objects, start=1):
                offsets.append(f.tell())
                f.write(f"{index} 0 obj\n".encode("latin-1"))
                f.write(obj)
                f.write(b"\nendobj\n")
            xref_offset = f.tell()
            f.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
            f.write(b"0000000000 65535 f \n")
            for offset in offsets[1:]:
                f.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
            f.write(
                (
                    f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                    f"startxref\n{xref_offset}\n%%EOF\n"
                ).encode("latin-1")
            )


def load_results(results_file: Path | None = None) -> dict[str, Any]:
    """Load the raw varying-prompt regret model-sweep results JSON."""
    path = results_file or RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No varying-prompt regret results found at {path}. "
            "Run paper_experiments/varying_prompt_regret_model_sweep.py before plotting."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(data).__name__}")
    return data


def _as_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_prompt_pairs(results: dict[str, Any], *, exclude_generation_errors: bool = False) -> list[PromptPair]:
    """Return one unique original/generated prompt pair per scenario repetition.

    The model sweep stores the same generated prompt once per evaluated model, so
    prompt rows are deduplicated by scenario plus prompt-generation seed when the
    seed is available, otherwise by scenario plus repetition, otherwise by the
    prompt text pair.
    """
    prompt_pairs: dict[tuple[Any, ...], PromptPair] = {}

    for model_results in results.values():
        if not isinstance(model_results, dict):
            continue
        for scenario, scenario_runs in model_results.items():
            if not isinstance(scenario_runs, list):
                continue
            for run in scenario_runs:
                if not isinstance(run, dict):
                    continue
                original_prompt = str(run.get("original_prompt") or TEST_PROMPTS.get(scenario, "")).strip()
                generated_prompt = str(run.get("prompt") or "").strip()
                if not original_prompt or not generated_prompt:
                    continue

                prompt_generation_error = run.get("prompt_generation_error")
                if exclude_generation_errors and prompt_generation_error:
                    continue

                seed = _as_optional_int(run.get("prompt_generation_seed"))
                rep = _as_optional_int(run.get("rep", run.get("seed")))
                if seed is not None:
                    dedupe_key = (scenario, "prompt_generation_seed", seed)
                elif rep is not None:
                    dedupe_key = (scenario, "rep", rep)
                else:
                    dedupe_key = (scenario, "text", original_prompt, generated_prompt)

                prompt_pairs.setdefault(
                    dedupe_key,
                    PromptPair(
                        scenario=scenario,
                        rep=rep,
                        prompt_generation_seed=seed,
                        original_prompt=original_prompt,
                        generated_prompt=generated_prompt,
                        prompt_generation_error=str(prompt_generation_error) if prompt_generation_error else None,
                    ),
                )

    return sorted(
        prompt_pairs.values(),
        key=lambda pair: (
            SCENARIO_ORDER.index(pair.scenario) if pair.scenario in SCENARIO_ORDER else len(SCENARIO_ORDER),
            pair.rep if pair.rep is not None else -1,
            pair.prompt_generation_seed if pair.prompt_generation_seed is not None else -1,
            pair.generated_prompt,
        ),
    )


def compute_distances(
    pairs: list[PromptPair],
    *,
    backend: Backend,
    sentence_transformer_model: str,
) -> tuple[list[DistanceRecord], str]:
    """Compute distances with the shared prompt-semantic-distance backend."""
    distances, backend_used = pair_distances(
        [(pair.original_prompt, pair.generated_prompt) for pair in pairs],
        backend=backend,
        model_name=sentence_transformer_model,
    )
    return [
        DistanceRecord(**pair.__dict__, distance=distance)
        for pair, distance in zip(pairs, distances, strict=True)
    ], backend_used


def summarise_distances(records: list[DistanceRecord]) -> list[dict[str, Any]]:
    """Aggregate distance records by scenario."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        values = [record.distance for record in records if record.scenario == scenario]
        if not values:
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "n_prompts": 0,
                    "average_distance": None,
                    "standard_deviation": None,
                    "minimum_distance": None,
                    "maximum_distance": None,
                }
            )
            continue
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "n_prompts": len(values),
                "average_distance": average(values),
                "standard_deviation": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "minimum_distance": min(values),
                "maximum_distance": max(values),
            }
        )
    return rows


def _format_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def write_summary_csv(summary_rows: list[dict[str, Any]], output_file: Path = SUMMARY_CSV) -> None:
    """Write scenario-level semantic-distance statistics as CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "scenario_label",
        "n_prompts",
        "average_distance",
        "standard_deviation",
        "minimum_distance",
        "maximum_distance",
    ]
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    **row,
                    "average_distance": _format_float(row["average_distance"]),
                    "standard_deviation": _format_float(row["standard_deviation"]),
                    "minimum_distance": _format_float(row["minimum_distance"]),
                    "maximum_distance": _format_float(row["maximum_distance"]),
                }
            )


def write_summary_markdown(summary_rows: list[dict[str, Any]], output_file: Path = SUMMARY_MD) -> None:
    """Write scenario-level semantic-distance statistics as Markdown."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Scenario | n prompts | Average distance | Std. dev. | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario_label"]),
                    str(row["n_prompts"]),
                    _format_float(row["average_distance"]),
                    _format_float(row["standard_deviation"]),
                    _format_float(row["minimum_distance"]),
                    _format_float(row["maximum_distance"]),
                ]
            )
            + " |"
        )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_details_csv(records: list[DistanceRecord], output_file: Path = DETAILS_CSV) -> None:
    """Write one row per deduplicated generated prompt."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "scenario_label",
        "rep",
        "prompt_generation_seed",
        "distance",
        "prompt_generation_error",
        "original_prompt",
        "generated_prompt",
    ]
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "scenario": record.scenario,
                    "scenario_label": SCENARIO_LABELS.get(record.scenario, record.scenario),
                    "rep": "" if record.rep is None else record.rep,
                    "prompt_generation_seed": ""
                    if record.prompt_generation_seed is None
                    else record.prompt_generation_seed,
                    "distance": _format_float(record.distance),
                    "prompt_generation_error": record.prompt_generation_error or "",
                    "original_prompt": record.original_prompt,
                    "generated_prompt": record.generated_prompt,
                }
            )


def nice_axis_max(value: float) -> float:
    """Return a readable upper bound for a plot axis."""
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return min(1.0, nice_fraction * (10**exponent))


def draw_combined_bar_plot(summary_rows: list[dict[str, Any]], output_file: Path = COMBINED_PLOT) -> None:
    """Draw one PDF bar chart with one average-distance bar per scenario."""
    page = PdfPage()
    page.text(54, 472, "Average semantic distance of generated prompts", size=16)

    plot_x = 82
    plot_y = 72
    plot_width = 580
    plot_height = 340
    values = [float(row["average_distance"] or 0.0) for row in summary_rows]
    y_max = nice_axis_max(max(values) * 1.15 if values else 1.0)

    page.line(plot_x, plot_y, plot_x, plot_y + plot_height, width=1.0)
    page.line(plot_x, plot_y, plot_x + plot_width, plot_y, width=1.0)

    for tick in range(6):
        value = y_max * tick / 5
        y = plot_y + (value / y_max) * plot_height
        page.line(plot_x - 4, y, plot_x, y, width=0.8)
        page.text(42, y - 4, f"{value:.2f}", size=8)
        if tick:
            page.line(plot_x, y, plot_x + plot_width, y, width=0.25, colour=(0.82, 0.82, 0.82))

    bar_slot = plot_width / len(summary_rows)
    bar_width = bar_slot * 0.56
    for index, row in enumerate(summary_rows):
        value = float(row["average_distance"] or 0.0)
        bar_height = (value / y_max) * plot_height if y_max else 0
        x = plot_x + index * bar_slot + (bar_slot - bar_width) / 2
        page.fill_rect(x, plot_y, bar_width, bar_height, SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)])
        page.text(x + 2, plot_y + bar_height + 8, f"{value:.3f}", size=8)
        label_words = str(row["scenario_label"]).split("-")
        for line_index, label in enumerate(label_words):
            page.text(x, plot_y - 18 - (line_index * 11), label, size=8)
        page.text(x + 4, plot_y - 45, f"n={row['n_prompts']}", size=7)

    page.text(258, 28, "Scenario", size=10)
    page.text(8, 248, "Cosine distance", size=10)
    page.save(output_file)


def draw_individual_bar_plots(summary_rows: list[dict[str, Any]], output_dir: Path = INDIVIDUAL_PLOT_DIR) -> None:
    """Draw one single-bar PDF chart for each scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_max = nice_axis_max(max(float(row["average_distance"] or 0.0) for row in summary_rows) * 1.15)

    for index, row in enumerate(summary_rows):
        page = PdfPage(width=420, height=360)
        label = str(row["scenario_label"])
        value = float(row["average_distance"] or 0.0)
        page.text(34, 326, f"{label}: average semantic distance", size=14)

        plot_x = 84
        plot_y = 62
        plot_width = 240
        plot_height = 220
        page.line(plot_x, plot_y, plot_x, plot_y + plot_height, width=1.0)
        page.line(plot_x, plot_y, plot_x + plot_width, plot_y, width=1.0)

        for tick in range(6):
            tick_value = y_max * tick / 5
            y = plot_y + (tick_value / y_max) * plot_height
            page.line(plot_x - 4, y, plot_x, y, width=0.8)
            page.text(44, y - 4, f"{tick_value:.2f}", size=8)
            if tick:
                page.line(plot_x, y, plot_x + plot_width, y, width=0.25, colour=(0.82, 0.82, 0.82))

        bar_width = 92
        bar_height = (value / y_max) * plot_height if y_max else 0
        x = plot_x + (plot_width - bar_width) / 2
        page.fill_rect(x, plot_y, bar_width, bar_height, SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)])
        page.text(x + 22, plot_y + bar_height + 10, f"{value:.3f}", size=11)
        page.text(x + 26, plot_y - 22, f"n={row['n_prompts']}", size=9)
        page.text(16, 182, "Cosine distance", size=10)
        page.save(output_dir / f"{row['scenario']}_semantic_distance.pdf")


def print_summary(summary_rows: list[dict[str, Any]], *, backend_used: str) -> None:
    """Print a compact text summary to stdout."""
    print(f"Semantic-distance backend: {backend_used}")
    print("Scenario              n prompts  avg distance  std dev")
    print("--------------------  ---------  ------------  -------")
    for row in summary_rows:
        print(
            f"{row['scenario_label']:<20}  {row['n_prompts']:>9}  "
            f"{_format_float(row['average_distance']):>12}  {_format_float(row['standard_deviation']):>7}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_FILE,
        help=f"Raw varying-prompt regret results JSON (default: {RESULTS_FILE}).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sentence-transformers", "tfidf"],
        default="auto",
        help="Distance backend. 'auto' tries sentence-transformers then falls back to TF-IDF.",
    )
    parser.add_argument(
        "--sentence-transformer-model",
        default=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        help="Sentence-transformers model name/path used when that backend is active.",
    )
    parser.add_argument(
        "--exclude-generation-errors",
        action="store_true",
        help="Exclude fallback prompts where prompt generation failed.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=SUMMARY_CSV,
        help=f"Scenario summary CSV path (default: {SUMMARY_CSV}).",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=SUMMARY_MD,
        help=f"Scenario summary Markdown path (default: {SUMMARY_MD}).",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=DETAILS_CSV,
        help=f"Per-prompt details CSV path (default: {DETAILS_CSV}).",
    )
    parser.add_argument(
        "--combined-plot",
        type=Path,
        default=COMBINED_PLOT,
        help=f"Combined six-bar PDF path (default: {COMBINED_PLOT}).",
    )
    parser.add_argument(
        "--individual-plot-dir",
        type=Path,
        default=INDIVIDUAL_PLOT_DIR,
        help=f"Directory for six individual bar-plot PDFs (default: {INDIVIDUAL_PLOT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    """Run semantic-distance aggregation and plotting."""
    args = parse_args()
    results = load_results(args.results_file)
    pairs = iter_prompt_pairs(results, exclude_generation_errors=args.exclude_generation_errors)
    if not pairs:
        raise ValueError(f"No prompt pairs found in {args.results_file}")

    distance_records, backend_used = compute_distances(
        pairs,
        backend=args.backend,
        sentence_transformer_model=args.sentence_transformer_model,
    )
    summary_rows = summarise_distances(distance_records)

    write_details_csv(distance_records, args.details_csv)
    write_summary_csv(summary_rows, args.summary_csv)
    write_summary_markdown(summary_rows, args.summary_md)
    draw_combined_bar_plot(summary_rows, args.combined_plot)
    draw_individual_bar_plots(summary_rows, args.individual_plot_dir)
    print_summary(summary_rows, backend_used=backend_used)

    print(f"\nWrote per-prompt distances to {args.details_csv}")
    print(f"Wrote summary table to {args.summary_csv} and {args.summary_md}")
    print(f"Wrote combined bar plot to {args.combined_plot}")
    print(f"Wrote six individual bar plots to {args.individual_plot_dir}")


if __name__ == "__main__":
    main()
