# Standalone attack-path planning demo

This folder contains a self-contained, synthetic example of the attack-path demo
as an importable Python module. It uses only standard-library Python and bundled
dummy data, so it can be run without a database, MCP server, optimiser, scanner,
or external security tools.

The example is defensive: the planning stub enumerates possible paths through a
toy enterprise graph, ranks them with a simple transparent risk heuristic,
prints mitigation actions that would break or reduce those paths, and writes a
small SVG visualisation of the highest-risk path.

## Run it

From the repository root:

```bash
python -m demo.attack_paths_demo.standalone_module.run_demo
```

The command prints ranked paths and writes an SVG to:

```text
demo/attack_paths_demo/standalone_module/outputs/attack_paths.svg
```

## Module layout

- `dummy_data.py` defines synthetic assets and graph transitions.
- `planning_stub.py` contains the reusable path-enumeration and ranking stub.
- `visualise.py` renders a dependency-free SVG of the ranked graph.
- `run_demo.py` is a small command-line entry point that also writes the SVG.

## Example use

```python
from demo.attack_paths_demo.standalone_module import (
    build_dummy_environment,
    plan_attack_paths,
)

environment = build_dummy_environment()
paths = plan_attack_paths(environment, max_depth=4, max_paths=3)
for path in paths:
    print(path.risk_score, path.assets, path.mitigations)
```
