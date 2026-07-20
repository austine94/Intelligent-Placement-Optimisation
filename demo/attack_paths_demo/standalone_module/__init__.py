"""Standalone attack-path planning demo with bundled dummy data."""

from .dummy_data import build_dummy_environment
from .planning_stub import plan_attack_paths
from .visualise import render_attack_path_svg

__all__ = ["build_dummy_environment", "plan_attack_paths", "render_attack_path_svg"]
