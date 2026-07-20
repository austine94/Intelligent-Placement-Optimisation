"""Planning stub for ranking synthetic attack paths.

This is deliberately a transparent heuristic rather than an offensive tool. It
helps demonstrate how a planning layer can identify defensive priorities from a
small graph of dummy assets and mitigations.
"""

from collections import defaultdict
from dataclasses import dataclass

from .dummy_data import AttackStep, DummyEnvironment


@dataclass(frozen=True)
class PlannedPath:
    """A ranked path from an entry point to a crown-jewel asset."""

    assets: tuple[str, ...]
    techniques: tuple[str, ...]
    mitigations: tuple[str, ...]
    risk_score: float


def _adjacency(steps: tuple[AttackStep, ...]) -> dict[str, list[AttackStep]]:
    graph: dict[str, list[AttackStep]] = defaultdict(list)
    for step in steps:
        graph[step.source].append(step)
    return graph


def _score_path(environment: DummyEnvironment, steps: tuple[AttackStep, ...]) -> float:
    likelihood = 1.0
    impact = 0.0
    for step in steps:
        likelihood *= step.likelihood
        impact += step.impact * environment.assets[step.target].business_value
    return round(likelihood * impact, 4)


def plan_attack_paths(
    environment: DummyEnvironment,
    max_depth: int = 4,
    max_paths: int = 5,
) -> list[PlannedPath]:
    """Enumerate and rank defensive planning paths through the dummy graph."""

    graph = _adjacency(environment.steps)
    completed: list[PlannedPath] = []
    stack: list[tuple[str, tuple[str, ...], tuple[AttackStep, ...]]] = [
        (entry, (entry,), ()) for entry in environment.entry_assets
    ]

    while stack:
        current, asset_path, step_path = stack.pop()
        if len(step_path) >= max_depth:
            continue
        for step in graph.get(current, []):
            if step.target in asset_path:
                continue
            next_assets = (*asset_path, step.target)
            next_steps = (*step_path, step)
            if step.target in environment.crown_jewels:
                completed.append(
                    PlannedPath(
                        assets=next_assets,
                        techniques=tuple(s.technique for s in next_steps),
                        mitigations=tuple(dict.fromkeys(s.mitigation for s in next_steps)),
                        risk_score=_score_path(environment, next_steps),
                    )
                )
            else:
                stack.append((step.target, next_assets, next_steps))

    return sorted(completed, key=lambda item: item.risk_score, reverse=True)[:max_paths]
