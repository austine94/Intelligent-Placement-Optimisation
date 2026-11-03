"""Regions, and UK land sampling, for the map view.

The optimisation runs on a unit square. A region maps that square onto a real
lon/lat window with equal ground distance on both axes, so unit-square distance
is proportional to real distance. Locations are generated on UK land only.
"""
import json
import math

import numpy as np
from matplotlib.path import Path as MplPath

from config import MAP_DATA, REGION_TXT, WORKLOAD_REGION_TXT

REGIONS = {
    "unit": {"label": "Unit square (abstract)"},
    "uk": {"label": "United Kingdom", "lon": (-10.7, 4.3), "lat": (50.0, 58.7), "scale_km": 200},
    "north-west": {"label": "North West England", "lon": (-4.0, -1.3), "lat": (53.2, 54.8), "scale_km": 25},
}
DEFAULT_REGION = "uk"


def _read(path):
    try:
        name = path.read_text().strip()
    except OSError:
        return None
    return name if name in REGIONS else None


def get_region():
    """Region chosen on the dashboard; applies to the next generated workload."""
    return _read(REGION_TXT) or DEFAULT_REGION


def set_region(name):
    if name not in REGIONS:
        raise ValueError(f"unknown region {name}")
    REGION_TXT.write_text(name + "\n")


def get_workload_region():
    """Region the current workload was generated in (what the plot must use)."""
    return _read(WORKLOAD_REGION_TXT) or get_region()


def set_workload_region(name):
    WORKLOAD_REGION_TXT.write_text(name + "\n")


class LandMask:
    def __init__(self, region_name):
        region = REGIONS[region_name]
        self.lon, self.lat = region["lon"], region["lat"]
        rings = json.loads(MAP_DATA.read_text())["uk_land"]
        self.paths = []
        for ring in rings:
            arr = np.array(ring)
            if arr[:, 0].max() < self.lon[0] or arr[:, 0].min() > self.lon[1]:
                continue
            if arr[:, 1].max() < self.lat[0] or arr[:, 1].min() > self.lat[1]:
                continue
            self.paths.append(MplPath(arr))

    def to_lonlat(self, pts):
        pts = np.atleast_2d(pts)
        return np.column_stack([self.lon[0] + pts[:, 0] * (self.lon[1] - self.lon[0]),
                                self.lat[0] + pts[:, 1] * (self.lat[1] - self.lat[0])])

    def on_land(self, pts):
        ll = self.to_lonlat(pts)
        inside = np.zeros(len(ll), dtype=bool)
        for path in self.paths:
            inside |= path.contains_points(ll)
        return inside

    def sample_uniform(self, k, rng=np.random):
        out = np.empty((0, 2))
        while len(out) < k:
            cand = rng.uniform(0, 1, size=(max(64, 4 * k), 2))
            out = np.vstack([out, cand[self.on_land(cand)]])
        return out[:k]

    def sample_near(self, mean, cov_var, rng=np.random, tries=200):
        """Gaussian draw around a hotspot, resampled until it lands on land."""
        cov = np.diag([cov_var, cov_var])
        for _ in range(tries):
            p = rng.multivariate_normal(mean, cov)
            if 0 <= p[0] <= 1 and 0 <= p[1] <= 1 and self.on_land(p[None, :])[0]:
                return p
        return self.sample_uniform(1, rng)[0]
