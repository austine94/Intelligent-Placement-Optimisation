# Standalone Agentic-AI Compute Provisioning Demo

This folder is a standalone demonstration of the proposed three-layer Agentic-AI system for compute provisioning. It is intentionally separate from the paper experiments, but it adapts their design patterns and optimisation objective.

## Architecture

The demo has three layers:

1. **Data ingestion layer**
   - `data_generator.py` generates a new workload using the repository's scenario-generation routines.
   - Raw viewer and compute-node data are written to `raw_data/viewers.csv` and `raw_data/compute_nodes.csv`.
   - The screening pipeline imputes missing costs and capacities with the average observed value, normalises costs onto `[0, 1]`, and loads the cleaned data into SQLite.

2. **MCP server layer**
   - `mcp_server.py` exposes the same experiment-style optimisation tools, plus the explicit tools described in the Agentic-AI system text:
     - `get_system_status`
     - `get_problem_status`
     - `solve_optimisation_problem`
     - `read_optimal_solution`
     - `read_policy_database`
     - `update_policy_database`
     - `render_optimal_solution`
   - The optimisation call uses the current repository's `flexmedia_delivery_instance.exact_solve` objective.

3. **SLM agent layer**
   - `agent.py` uses Ollama for structured JSON action selection.
   - The workflow is deterministic and validated to reduce unstable tool calls and infinite ReAct-style loops.
   - If the selected action is malformed or unexpected, the agent replaces it with the expected next workflow step and records that validation event.
   - If no optimisation tool is called within the retry budget, the agent emits a failure flag.

## Quick start

From the repository root:

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo/requirements.txt
ollama serve
ollama pull llama3.2
python demo/run_demo.py --model llama3.2 --policy "minimise financial cost"
```

In another terminal, launch the browser dashboard:

```bash
python demo/frontend.py
```

Then open <http://127.0.0.1:8765>.

To run a changing workload every 30 seconds:

```bash
python demo/run_demo.py --loop --interval 30 --model llama3.2
```

## Changing the operational priority during runtime

The active priority is stored as a single policy row in SQLite and mirrored in `policy.txt`. You can update it by using the MCP tool `update_policy_database`, or directly before launch:

```bash
python demo/run_demo.py --policy "minimise energy usage"
python demo/run_demo.py --policy "minimise QoE latency"
python demo/run_demo.py --policy "balanced cost energy and latency trade-off"
```

The demo maps the natural-language policy to the experiment objective parameters:

| Priority wording | alpha | beta | Meaning |
| --- | ---: | ---: | --- |
| financial / cost | 0.0 | 1.0 | minimise financial cost |
| energy | 0.0 | 0.0 | minimise energy-weighted delivery cost |
| latency / QoE / quality | 1.0 | 0.5 | prioritise latency/QoE |
| balanced / trade-off | 0.1 | 0.5 | mixed cost-energy-latency objective |

## Browser dashboard

`frontend.py` provides a lightweight browser dashboard without adding another web framework dependency. It reads directly from the SQLite database and refreshes every five seconds. The dashboard shows ingestion status, current policy, latest optimisation metrics, open compute nodes, the rendered allocation plot, and the raw latest solution JSON. You can also update the active operational priority from the dashboard while the demo is running; the next agent cycle will read it through the MCP policy tool.

```bash
python demo/frontend.py --host 127.0.0.1 --port 8765
```

## Outputs

Each cycle produces:

- fresh raw CSVs in `demo/raw_data/`;
- cleaned SQLite tables in `demo/agentic_demo.sqlite`;
- a new row in `optimal_solutions`;
- a human-readable report in `demo/outputs/latest_solution.md`;
- a visual allocation plot in `demo/outputs/latest_solution.png` if Matplotlib is available.

## Notes for reviewers

- SQLite is used instead of Postgres to keep the demo portable.
- The MCP server is real and runs over stdio through FastMCP.
- Ollama is used for the SLM layer, but the workflow remains deterministic and validated so that reviewers can reproduce behaviour.
- The optimisation is intentionally delegated to the existing repository optimisation engine, so the demo follows the same mathematical objective as the experiments.
