"""Utilities for normalising generated FlexMedia problem data."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _normalise_array(values: np.ndarray) -> np.ndarray:
    """Return a min-max normalised copy of ``values`` on the interval [0, 1]."""
    values = values.astype(float, copy=True)
    min_value = float(np.min(values))
    max_value = float(np.max(values))

    if max_value == min_value:
        return np.zeros_like(values, dtype=float)

    return (values - min_value) / (max_value - min_value)


def _normalise_joint_arrays(arrays: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Min-max normalise arrays using a shared scaler across all values."""
    arrays = [array.astype(float, copy=True) for array in arrays]
    min_value = min(float(np.min(array)) for array in arrays)
    max_value = max(float(np.max(array)) for array in arrays)

    if max_value == min_value:
        return [np.zeros_like(array, dtype=float) for array in arrays]

    return [(array - min_value) / (max_value - min_value) for array in arrays]


def normalise_generated_problem_data(data: dict[str, object]) -> dict[str, object]:
    """Normalise generated costs, energy costs, and latencies onto [0, 1].

    Financial costs are normalised jointly so datacentre opening costs and
    consumer delivery costs remain on a shared scale. Energy costs and latency
    values are normalised independently.
    """
    normalised_data = data.copy()

    datacentre_costs = np.array(data["datacentre_costs"], dtype=float)
    consumer_costs = np.array(data["consumer_costs"], dtype=float)
    datacentre_costs, consumer_costs = _normalise_joint_arrays(
        [datacentre_costs, consumer_costs]
    )

    normalised_data["datacentre_costs"] = datacentre_costs.tolist()
    normalised_data["consumer_costs"] = consumer_costs.tolist()
    normalised_data["energy_costs"] = _normalise_array(
        np.array(data["energy_costs"], dtype=float)
    ).tolist()
    normalised_data["consumer_latencies"] = _normalise_array(
        np.array(data["consumer_latencies"], dtype=float)
    ).tolist()

    return normalised_data
