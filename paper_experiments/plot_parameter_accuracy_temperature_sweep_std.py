#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot parameter-selection score standard deviation by temperature.

This post-processing script reads the CSV table produced by
``parameter_accuracy_selection_std_tables.py`` for the temperature sweep and
writes a paper-ready PDF line chart with one line per scenario. It does not
rerun any Ollama models.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_accuracy_selection_std_tables import TEMPERATURE_SWEEP_STD_TABLE_CSV  # noqa: E402
from plot_parameter_accuracy_temperature_sweep import PdfPage, SCENARIO_COLOURS  # noqa: E402

PLOTS_DIR = SCRIPT_DIR / "paper_plots"
OUTPUT_FILE = PLOTS_DIR / "parameter_accuracy_temperature_sweep_standard_deviation_by_scenario.pdf"


def _as_float(value: str | None) -> float | None:
    """Return a CSV value as float when possible and non-empty."""
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_standard_deviation_table(
    csv_file: Path | None = None,
) -> dict[str, list[tuple[float, float | None]]]:
    """Load scenario standard-deviation series from the temperature-sweep CSV table."""
    path = csv_file or TEMPERATURE_SWEEP_STD_TABLE_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No parameter-selection standard-deviation table found at {path}. "
            "Run paper_experiments/parameter_accuracy_selection_std_tables.py before plotting."
        )

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "scenario" not in reader.fieldnames:
            raise ValueError(f"Expected a CSV table with a 'scenario' column in {path}.")

        temperature_columns = [column for column in reader.fieldnames if column != "scenario"]
        temperatures: list[tuple[str, float]] = []
        for column in temperature_columns:
            temperature = _as_float(column)
            if temperature is not None:
                temperatures.append((column, temperature))

        if not temperatures:
            raise ValueError(f"Expected at least one numeric temperature column in {path}.")

        series: dict[str, list[tuple[float, float | None]]] = {}
        for row in reader:
            scenario = row.get("scenario", "").strip()
            if not scenario:
                continue
            series[scenario] = [
                (temperature, _as_float(row.get(column, "")))
                for column, temperature in temperatures
            ]

    if not series:
        raise ValueError(f"No scenario rows found in {path}.")
    return series


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


def draw_standard_deviation_plot(
    series: dict[str, list[tuple[float, float | None]]],
    output_file: Path = OUTPUT_FILE,
) -> None:
    """Draw the standard-deviation line plot with one line per scenario."""
    if not series:
        raise ValueError("No standard-deviation values found to plot.")

    page = PdfPage()
    page.text(54, 472, "Parameter-selection standard deviation by temperature", size=16)

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

    finite_values = [value for values in series.values() for _, value in values if value is not None]
    y_max = _nice_axis_max(max(finite_values) * 1.08) if finite_values else 0.1

    # Axes and grid.
    page.line(plot_x, plot_y, plot_x + plot_width, plot_y, width=1.0)
    page.line(plot_x, plot_y, plot_x, plot_y + plot_height, width=1.0)
    page.text(plot_x + plot_width / 2 - 34, 30, "Temperature", size=12)
    page.text(18, plot_y + plot_height / 2, "Std. dev.", size=12)

    for tick in range(0, 6):
        value = y_max * tick / 5
        tick_y = plot_y + (value / y_max) * plot_height
        page.line(plot_x - 4, tick_y, plot_x, tick_y, width=0.75)
        page.text(plot_x - 42, tick_y - 3, f"{value:.2f}", size=10)
        if tick:
            page.line(plot_x, tick_y, plot_x + plot_width, tick_y, width=0.25, colour=(0.82, 0.82, 0.82))

    for temperature in temperatures:
        tick_x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        page.line(tick_x, plot_y, tick_x, plot_y - 4, width=0.75)
        label = f"{temperature:g}"
        page.text(tick_x - 5 * len(label) / 2, plot_y - 18, label, size=10)

    def map_point(temperature: float, value: float) -> tuple[float, float]:
        x = plot_x + ((temperature - x_min) / (x_max - x_min)) * plot_width
        y = plot_y + (value / y_max) * plot_height
        return x, y

    for index, (scenario, values) in enumerate(series.items()):
        colour = SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)]
        points = [map_point(temperature, value) for temperature, value in values if value is not None]
        page.polyline(points, width=1.35, colour=colour)
        for x, y in points:
            page.marker(x, y, colour)

    legend_x = 604
    legend_y = 394
    page.text(legend_x, legend_y + 20, "Scenario", size=13)
    for index, scenario in enumerate(series):
        y = legend_y - index * 20
        colour = SCENARIO_COLOURS[index % len(SCENARIO_COLOURS)]
        page.line(legend_x, y, legend_x + 22, y, width=1.4, colour=colour)
        page.marker(legend_x + 11, y, colour)
        page.text(legend_x + 30, y - 4, scenario, size=10)

    page.save(output_file)
    print(f"Saved plot: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot parameter-selection standard deviation by temperature with one line per scenario."
    )
    parser.add_argument(
        "--csv-file",
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
    """Load the standard-deviation table and write the PDF figure."""
    args = parse_args()
    series = load_standard_deviation_table(args.csv_file)
    draw_standard_deviation_plot(series, args.output_file)


if __name__ == "__main__":
    main()
