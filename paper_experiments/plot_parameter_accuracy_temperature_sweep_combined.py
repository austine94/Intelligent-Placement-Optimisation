#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot parameter-selection accuracy and standard deviation by temperature.

This post-processing script combines the mean accuracy JSON produced by
``parameter_accuracy_temperature_sweep.py`` with the standard-deviation CSV
produced by ``parameter_accuracy_selection_std_tables.py``. It writes one
paper-ready PDF with mean selection accuracy on the left y-axis, standard
deviation on the right y-axis, and one colour per scenario. Mean lines are
solid; standard-deviation lines are dashed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_accuracy_selection_std_tables import TEMPERATURE_SWEEP_STD_TABLE_CSV  # noqa: E402
from parameter_accuracy_temperature_sweep import RESULTS_FILE  # noqa: E402
from plot_parameter_accuracy_temperature_sweep import (  # noqa: E402
    PdfPage,
    SCENARIO_COLOURS,
    aggregate_accuracy_by_scenario_temperature,
    load_results,
)
from plot_parameter_accuracy_temperature_sweep_std import load_standard_deviation_table  # noqa: E402

PLOTS_DIR = SCRIPT_DIR / "paper_plots"
OUTPUT_FILE = PLOTS_DIR / "parameter_accuracy_temperature_sweep_combined_by_scenario.pdf"

Series = dict[str, list[tuple[float, float | None]]]


def _nice_axis_max(value: float) -> float:
    """Return a readable upper bound for a plot axis."""
    if value <= 0:
        return 0.1
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
    return nice_fraction * (10**exponent)


def _series_lookup(series: Series, scenario: str, temperature: float) -> float | None:
    """Return a series value for a scenario/temperature pair when present."""
    for point_temperature, value in series.get(scenario, []):
        if math.isclose(point_temperature, temperature):
            return value
    return None


def _dashed_polyline(
    page: PdfPage,
    points: list[tuple[float, float]],
    width: float = 1.35,
    colour: tuple[float, float, float] = (0, 0, 0),
) -> None:
    """Draw a dashed polyline on the small PDF canvas."""
    if len(points) < 2:
        return
    r, g, b = colour
    commands = [f"{r:.2f} {g:.2f} {b:.2f} RG {width:.2f} w [4 3] 0 d"]
    start_x, start_y = points[0]
    commands.append(f"{start_x:.2f} {start_y:.2f} m")
    for x, y in points[1:]:
        commands.append(f"{x:.2f} {y:.2f} l")
    commands.append("S [] 0 d")
    page.commands.append(" ".join(commands))


def _dashed_line(
    page: PdfPage,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float = 1.35,
    colour: tuple[float, float, float] = (0, 0, 0),
) -> None:
    """Draw a dashed line segment on the small PDF canvas."""
    r, g, b = colour
    page.commands.append(
        f"{r:.2f} {g:.2f} {b:.2f} RG {width:.2f} w [4 3] 0 d "
        f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S [] 0 d"
    )


def draw_combined_plot(
    accuracy_series: Series,
    std_series: Series,
    output_file: Path = OUTPUT_FILE,
) -> None:
    """Draw a dual-axis plot of mean accuracy and standard deviation."""
    if not accuracy_series:
        raise ValueError("No mean accuracy values found to plot.")
    if not std_series:
        raise ValueError("No standard-deviation values found to plot.")

    scenario_order = [scenario for scenario in accuracy_series if scenario in std_series]
    scenario_order.extend(scenario for scenario in accuracy_series if scenario not in scenario_order)
    scenario_order.extend(scenario for scenario in std_series if scenario not in scenario_order)

    temperatures = sorted(
        {
            temperature
            for series in (accuracy_series, std_series)
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
    std_values = [value for values in std_series.values() for _, value in values if value is not None]
    right_y_max = _nice_axis_max(max(std_values) * 1.08) if std_values else 0.1

    page = PdfPage(width=820, height=528)
    page.text(54, 496, "Parameter-selection accuracy and variability by temperature", size=16)

    plot_x = 82
    plot_y = 74
    plot_width = 520
    plot_height = 360
    right_axis_x = plot_x + plot_width

    # Axes.
    page.line(plot_x, plot_y, right_axis_x, plot_y, width=1.0)
    page.line(plot_x, plot_y, plot_x, plot_y + plot_height, width=1.0)
    page.line(right_axis_x, plot_y, right_axis_x, plot_y + plot_height, width=1.0)
    page.text(plot_x + plot_width / 2 - 34, 36, "Temperature", size=10)
    page.text(18, plot_y + plot_height / 2, "Mean accuracy", size=10)
    page.text(right_axis_x + 56, plot_y + plot_height / 2, "Std. dev.", size=10)

    for tick in range(0, 6):
        left_value = left_y_max * tick / 5
        tick_y = plot_y + (left_value / left_y_max) * plot_height
        right_value = right_y_max * tick / 5

        page.line(plot_x - 4, tick_y, plot_x, tick_y, width=0.75)
        page.text(plot_x - 36, tick_y - 3, f"{left_value:.1f}", size=8)

        page.line(right_axis_x, tick_y, right_axis_x + 4, tick_y, width=0.75)
        page.text(right_axis_x + 8, tick_y - 3, f"{right_value:.2f}", size=8)

        if tick:
            page.line(plot_x, tick_y, right_axis_x, tick_y, width=0.25, colour=(0.82, 0.82, 0.82))

    for temperature in temperatures:
        tick_x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        page.line(tick_x, plot_y, tick_x, plot_y - 4, width=0.75)
        label = f"{temperature:g}"
        page.text(tick_x - 5 * len(label) / 2, plot_y - 18, label, size=8)

    def map_accuracy_point(temperature: float, value: float) -> tuple[float, float]:
        x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        y = plot_y + (value / left_y_max) * plot_height
        return x, y

    def map_std_point(temperature: float, value: float) -> tuple[float, float]:
        x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        y = plot_y + (value / right_y_max) * plot_height
        return x, y

    for index, scenario in enumerate(scenario_order):
        colour = SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)]
        accuracy_points = [
            map_accuracy_point(temperature, value)
            for temperature in temperatures
            if (value := _series_lookup(accuracy_series, scenario, temperature)) is not None
        ]
        std_points = [
            map_std_point(temperature, value)
            for temperature in temperatures
            if (value := _series_lookup(std_series, scenario, temperature)) is not None
        ]

        page.polyline(accuracy_points, width=1.35, colour=colour)
        _dashed_polyline(page, std_points, width=1.35, colour=colour)
        for x, y in accuracy_points:
            page.marker(x, y, colour)
        for x, y in std_points:
            page.marker(x, y, colour, size=2.4)

    # Scenario colour legend.
    legend_x = 626
    legend_y = 414
    page.text(legend_x, legend_y + 20, "Scenario", size=11)
    for index, scenario in enumerate(scenario_order):
        y = legend_y - index * 20
        colour = SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)]
        page.line(legend_x, y, legend_x + 22, y, width=1.4, colour=colour)
        page.marker(legend_x + 11, y, colour)
        page.text(legend_x + 30, y - 4, scenario, size=8)

    # Line-style legend.
    style_y = legend_y - len(scenario_order) * 20 - 18
    page.text(legend_x, style_y + 20, "Line style", size=11)
    page.line(legend_x, style_y, legend_x + 22, style_y, width=1.4)
    page.text(legend_x + 30, style_y - 4, "Mean accuracy", size=8)
    _dashed_line(page, legend_x, style_y - 20, legend_x + 22, style_y - 20, width=1.4)
    page.text(legend_x + 30, style_y - 24, "Std. dev.", size=8)

    page.save(output_file)
    print(f"Saved plot: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean parameter-selection accuracy and standard deviation by "
            "temperature with one colour per scenario."
        )
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_FILE,
        help=f"Raw accuracy JSON results to plot (default: {RESULTS_FILE})",
    )
    parser.add_argument(
        "--std-csv-file",
        type=Path,
        default=TEMPERATURE_SWEEP_STD_TABLE_CSV,
        help=f"Standard-deviation CSV table to plot (default: {TEMPERATURE_SWEEP_STD_TABLE_CSV})",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_FILE,
        help=f"PDF output path (default: {OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    """Load mean and standard-deviation inputs, then write the combined PDF."""
    args = parse_args()
    accuracy_series = aggregate_accuracy_by_scenario_temperature(load_results(args.results_file))
    std_series = load_standard_deviation_table(args.std_csv_file)
    draw_combined_plot(accuracy_series, std_series, args.output_file)


if __name__ == "__main__":
    main()
