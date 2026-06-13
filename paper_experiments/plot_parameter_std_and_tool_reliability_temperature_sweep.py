#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot parameter-selection variability and tool-call reliability by temperature.

This post-processing script combines the parameter-selection standard-deviation
CSV produced by ``parameter_accuracy_selection_std_tables.py`` with the raw JSON
produced by ``tool_calling_reliability_temperature_sweep.py``. It writes one
paper-ready PDF with tool-call success proportion on the left y-axis, parameter
selection score standard deviation on the right y-axis, and one colour per
scenario. Both y-axes use the same 0 to 1.0 range. Reliability lines are solid;
standard-deviation lines are dashed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_accuracy_selection_std_tables import TEMPERATURE_SWEEP_STD_TABLE_CSV  # noqa: E402
from parameter_selection_accuracy import SCENARIO_LABELS, SCENARIO_ORDER  # noqa: E402
from plot_parameter_accuracy_temperature_sweep import PdfPage, SCENARIO_COLOURS  # noqa: E402
from plot_parameter_accuracy_temperature_sweep_combined import (  # noqa: E402
    Series,
    _dashed_polyline,
    _series_lookup,
)
from plot_parameter_accuracy_temperature_sweep_std import load_standard_deviation_table  # noqa: E402
from tool_calling_reliability import average  # noqa: E402
from tool_calling_reliability_temperature_sweep import RESULTS_FILE as RELIABILITY_RESULTS_FILE  # noqa: E402

PLOTS_DIR = SCRIPT_DIR / "paper_plots"
OUTPUT_FILE = PLOTS_DIR / "parameter_std_and_tool_reliability_temperature_sweep_by_scenario.pdf"


def load_reliability_results(path: Path = RELIABILITY_RESULTS_FILE) -> dict[str, Any]:
    """Load raw tool-calling reliability temperature-sweep JSON results."""
    if not path.exists():
        raise FileNotFoundError(
            f"No tool-calling reliability temperature-sweep results found at {path}. "
            "Run paper_experiments/tool_calling_reliability_temperature_sweep.py before plotting."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(data).__name__}")
    return data


def aggregate_tool_success_by_scenario_temperature(results: dict[str, Any]) -> Series:
    """Return mean correct-tool-call proportions by scenario label and temperature."""
    series: Series = {}

    for scenario in SCENARIO_ORDER:
        scenario_label = SCENARIO_LABELS[scenario]
        points: list[tuple[float, float | None]] = []

        for temperature_key in sorted(results, key=lambda value: float(value)):
            try:
                temperature = float(temperature_key)
            except ValueError:
                continue

            runs = results.get(temperature_key, {}).get(scenario, [])
            if not isinstance(runs, list):
                points.append((temperature, None))
                continue

            successes = [
                float(run["correct_tool_called"])
                for run in runs
                if isinstance(run, dict) and run.get("correct_tool_called") is not None
            ]
            points.append((temperature, average(successes)))

        if points:
            series[scenario_label] = points

    return series


def draw_parameter_std_and_reliability_plot(
    reliability_series: Series,
    std_series: Series,
    output_file: Path = OUTPUT_FILE,
) -> None:
    """Draw a dual-axis plot of tool-call reliability and parameter-selection std dev."""
    if not reliability_series:
        raise ValueError("No tool-call reliability values found to plot.")
    if not std_series:
        raise ValueError("No parameter-selection standard-deviation values found to plot.")

    scenario_order = [scenario for scenario in reliability_series if scenario in std_series]
    scenario_order.extend(scenario for scenario in reliability_series if scenario not in scenario_order)
    scenario_order.extend(scenario for scenario in std_series if scenario not in scenario_order)

    temperatures = sorted(
        {
            temperature
            for series in (reliability_series, std_series)
            for values in series.values()
            for temperature, _ in values
        }
    )
    if not temperatures:
        raise ValueError("No temperatures found to plot.")

    x_min = min(temperatures)
    x_max = max(temperatures)
    if math.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    left_y_max = 1.0
    right_y_max = 1.0

    page = PdfPage(width=860, height=528)
    page.text(54, 496, "Tool-call reliability and parameter variability by temperature", size=16)

    plot_x = 86
    plot_y = 74
    plot_width = 520
    plot_height = 360
    right_axis_x = plot_x + plot_width

    # Axes.
    page.line(plot_x, plot_y, right_axis_x, plot_y, width=1.0)
    page.line(plot_x, plot_y, plot_x, plot_y + plot_height, width=1.0)
    page.line(right_axis_x, plot_y, right_axis_x, plot_y + plot_height, width=1.0)
    page.text(plot_x + plot_width / 2 - 34, 36, "Temperature", size=16)
    page.text(14, plot_y + plot_height / 2, "Tool success", size=16)
    page.text(right_axis_x + 50, plot_y + plot_height / 2, "Param. std. dev.", size=16)

    for tick in range(0, 6):
        left_value = left_y_max * tick / 5
        tick_y = plot_y + (left_value / left_y_max) * plot_height
        right_value = right_y_max * tick / 5

        page.line(plot_x - 4, tick_y, plot_x, tick_y, width=0.75)
        page.text(plot_x - 36, tick_y - 3, f"{left_value:.1f}", size=16)

        page.line(right_axis_x, tick_y, right_axis_x + 4, tick_y, width=0.75)
        page.text(right_axis_x + 8, tick_y - 3, f"{right_value:.2f}", size=16)

        if tick:
            page.line(plot_x, tick_y, right_axis_x, tick_y, width=0.25, colour=(0.82, 0.82, 0.82))

    for temperature in temperatures:
        tick_x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        page.line(tick_x, plot_y, tick_x, plot_y - 4, width=0.75)
        label = f"{temperature:g}"
        page.text(tick_x - 5 * len(label) / 2, plot_y - 18, label, size=16)

    def map_reliability_point(temperature: float, value: float) -> tuple[float, float]:
        x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        y = plot_y + (value / left_y_max) * plot_height
        return x, y

    def map_std_point(temperature: float, value: float) -> tuple[float, float]:
        x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        y = plot_y + (value / right_y_max) * plot_height
        return x, y

    for index, scenario in enumerate(scenario_order):
        colour = SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)]
        reliability_points = [
            map_reliability_point(temperature, value)
            for temperature in temperatures
            if (value := _series_lookup(reliability_series, scenario, temperature)) is not None
        ]
        std_points = [
            map_std_point(temperature, value)
            for temperature in temperatures
            if (value := _series_lookup(std_series, scenario, temperature)) is not None
        ]

        page.polyline(reliability_points, width=1.35, colour=colour)
        _dashed_polyline(page, std_points, width=1.35, colour=colour)
        for x, y in reliability_points:
            page.marker(x, y, colour)
        for x, y in std_points:
            page.marker(x, y, colour, size=2.4)

    # Scenario colour legend.
    legend_x = 630
    legend_y = 414
    page.text(legend_x, legend_y + 20, "Scenario", size=16)
    for index, scenario in enumerate(scenario_order):
        y = legend_y - index * 20
        colour = SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)]
        page.line(legend_x, y, legend_x + 22, y, width=1.4, colour=colour)
        page.marker(legend_x + 11, y, colour)
        page.text(legend_x + 30, y - 4, scenario, size=16)

    # Line-style key.
    key_x = 630
    key_y = 268
    page.text(key_x, key_y + 20, "Line", size=16)
    page.line(key_x, key_y, key_x + 30, key_y, width=1.35, colour=(0.0, 0.0, 0.0))
    page.text(key_x + 38, key_y - 4, "Tool success", size=16)
    _dashed_polyline(page, [(key_x, key_y - 22), (key_x + 30, key_y - 22)], width=1.35)
    page.text(key_x + 38, key_y - 26, "Param. std.", size=16)

    page.save(output_file)
    print(f"Saved plot: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot tool-call reliability and parameter-selection standard deviation "
            "by temperature with one colour per scenario."
        )
    )
    parser.add_argument(
        "--reliability-results-file",
        type=Path,
        default=RELIABILITY_RESULTS_FILE,
        help=f"Raw tool reliability JSON results to plot (default: {RELIABILITY_RESULTS_FILE})",
    )
    parser.add_argument(
        "--std-csv-file",
        type=Path,
        default=TEMPERATURE_SWEEP_STD_TABLE_CSV,
        help=f"Parameter-selection standard-deviation CSV table to plot (default: {TEMPERATURE_SWEEP_STD_TABLE_CSV})",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_FILE,
        help=f"PDF output path (default: {OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    """Load reliability and standard-deviation inputs, then write the combined PDF."""
    args = parse_args()
    reliability_series = aggregate_tool_success_by_scenario_temperature(
        load_reliability_results(args.reliability_results_file)
    )
    std_series = load_standard_deviation_table(args.std_csv_file)
    draw_parameter_std_and_reliability_plot(reliability_series, std_series, args.output_file)


if __name__ == "__main__":
    main()
