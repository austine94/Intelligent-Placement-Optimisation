#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot parameter-selection accuracy by temperature with one line per model.

This post-processing script reads the raw JSON produced by
``parameter_accuracy_temperature_sweep.py`` and writes a paper-ready PDF line
chart. It does not rerun any Ollama models.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_accuracy_temperature_sweep import MODEL, RESULTS_FILE, TEMPERATURES  # noqa: E402
from parameter_selection_accuracy import average  # noqa: E402

PLOTS_DIR = SCRIPT_DIR / "paper_plots"
OUTPUT_FILE = PLOTS_DIR / "parameter_accuracy_temperature_sweep_by_model.pdf"

MODEL_COLOURS: list[tuple[float, float, float]] = [
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
]


class PdfPage:
    """Tiny PDF writer with enough drawing primitives for a line chart."""

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

    def polyline(
        self,
        points: list[tuple[float, float]],
        width: float = 1.35,
        colour: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        if len(points) < 2:
            return
        r, g, b = colour
        commands = [f"{r:.2f} {g:.2f} {b:.2f} RG {width:.2f} w"]
        start_x, start_y = points[0]
        commands.append(f"{start_x:.2f} {start_y:.2f} m")
        for x, y in points[1:]:
            commands.append(f"{x:.2f} {y:.2f} l")
        commands.append("S")
        self.commands.append(" ".join(commands))

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

    def marker(self, x: float, y: float, colour: tuple[float, float, float], size: float = 3.2) -> None:
        self.fill_rect(x - size / 2, y - size / 2, size, size, colour)

    def save(self, output_file: Path) -> None:
        """Write the page to a single-page PDF file."""
        content = "\n".join(self.commands).encode("latin-1")
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
    """Load raw parameter-accuracy temperature-sweep results."""
    path = results_file or RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No parameter-accuracy temperature-sweep results found at {path}. "
            "Run paper_experiments/parameter_accuracy_temperature_sweep.py before plotting."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(data).__name__}")
    return data


def _as_float(value: Any) -> float | None:
    """Return value as float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_scores(
    destination: dict[str, dict[float, list[float]]],
    model: str,
    temperature: float,
    scenario_runs: Any,
) -> None:
    """Append parameter-selection scores from one scenario run list."""
    if not isinstance(scenario_runs, list):
        return

    for run in scenario_runs:
        if not isinstance(run, dict):
            continue
        score = _as_float(run.get("parameter_selection_score"))
        if score is None:
            continue
        run_model = str(run.get("model") or model or MODEL)
        run_temperature = _as_float(run.get("temperature"))
        destination[run_model][run_temperature if run_temperature is not None else temperature].append(score)


def aggregate_accuracy_by_model_temperature(results: dict[str, Any]) -> dict[str, list[tuple[float, float | None]]]:
    """Aggregate mean parameter-selection accuracy for each model and temperature.

    The current temperature-sweep JSON is keyed as ``temperature -> scenario -> runs``
    and contains one model. This function also accepts richer files keyed as
    ``model -> temperature -> scenario -> runs`` or files with ``model`` stored on
    each record, so the plotting script remains useful if the sweep is expanded
    to multiple models.
    """
    scores: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for outer_key, outer_value in results.items():
        outer_temperature = _as_float(outer_key)
        if outer_temperature is not None and isinstance(outer_value, dict):
            for scenario_runs in outer_value.values():
                _collect_scores(scores, MODEL, outer_temperature, scenario_runs)
            continue

        if not isinstance(outer_value, dict):
            continue

        model = str(outer_key)
        for temp_key, scenarios in outer_value.items():
            temperature = _as_float(temp_key)
            if temperature is None or not isinstance(scenarios, dict):
                continue
            for scenario_runs in scenarios.values():
                _collect_scores(scores, model, temperature, scenario_runs)

    all_temperatures = sorted({temp for model_scores in scores.values() for temp in model_scores})
    if not all_temperatures:
        all_temperatures = [float(temperature) for temperature in TEMPERATURES]

    series: dict[str, list[tuple[float, float | None]]] = {}
    for model, model_scores in scores.items():
        series[model] = [(temperature, average(model_scores.get(temperature, []))) for temperature in all_temperatures]

    return series


def draw_temperature_accuracy_plot(
    series: dict[str, list[tuple[float, float | None]]],
    output_file: Path = OUTPUT_FILE,
) -> None:
    """Draw the line plot with temperature on x-axis and one line per model."""
    if not series:
        raise ValueError("No parameter-selection scores found to plot.")

    page = PdfPage()
    page.text(54, 472, "Parameter-selection accuracy by temperature", size=16)

    plot_x = 78
    plot_y = 68
    plot_width = 510
    plot_height = 348
    temperatures = sorted({temperature for values in series.values() for temperature, _ in values})
    x_min = min(temperatures)
    x_max = max(temperatures)
    if math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    y_max = 1.0

    # Axes and grid.
    page.line(plot_x, plot_y, plot_x + plot_width, plot_y, width=1.0)
    page.line(plot_x, plot_y, plot_x, plot_y + plot_height, width=1.0)
    page.text(plot_x + plot_width / 2 - 34, 30, "Temperature", size=10)
    page.text(18, plot_y + plot_height / 2, "Mean accuracy", size=10)

    for tick in range(0, 6):
        value = y_max * tick / 5
        tick_y = plot_y + (value / y_max) * plot_height
        page.line(plot_x - 4, tick_y, plot_x, tick_y, width=0.75)
        page.text(plot_x - 36, tick_y - 3, f"{value:.1f}", size=8)
        if tick:
            page.line(plot_x, tick_y, plot_x + plot_width, tick_y, width=0.25, colour=(0.82, 0.82, 0.82))

    for temperature in temperatures:
        tick_x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        page.line(tick_x, plot_y, tick_x, plot_y - 4, width=0.75)
        label = f"{temperature:g}"
        page.text(tick_x - 5 * len(label) / 2, plot_y - 18, label, size=8)

    def map_point(temperature: float, value: float) -> tuple[float, float]:
        x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        y = plot_y + (value / y_max) * plot_height
        return x, y

    for index, (model, values) in enumerate(series.items()):
        colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]
        points = [map_point(temperature, value) for temperature, value in values if value is not None]
        page.polyline(points, width=1.35, colour=colour)
        for x, y in points:
            page.marker(x, y, colour)

    legend_x = 604
    legend_y = 394
    page.text(legend_x, legend_y + 20, "Model", size=11)
    for index, model in enumerate(series):
        y = legend_y - index * 20
        colour = MODEL_COLOURS[index % len(MODEL_COLOURS)]
        page.line(legend_x, y, legend_x + 22, y, width=1.4, colour=colour)
        page.marker(legend_x + 11, y, colour)
        page.text(legend_x + 30, y - 4, model, size=8)

    page.save(output_file)
    print(f"Saved plot: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot parameter-selection accuracy by temperature with one line per model."
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_FILE,
        help=f"Raw JSON results to plot (default: {RESULTS_FILE})",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_FILE,
        help=f"PDF output path (default: {OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    """Load results, aggregate scores, and write the PDF figure."""
    args = parse_args()
    results = load_results(args.results_file)
    series = aggregate_accuracy_by_model_temperature(results)
    draw_temperature_accuracy_plot(series, args.output_file)


if __name__ == "__main__":
    main()
