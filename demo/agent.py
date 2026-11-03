import argparse
import asyncio
import json
import math
import random
import subprocess
import urllib.request
from pathlib import Path

from config import BUCKETS_TXT, DEFAULT_BUCKETS, DEFAULT_INTERVAL_SECONDS
from data_generator import generate_snapshot
from database import initialise_database, set_policy

WORKFLOW = [
    {"tool": "get_system_status", "arguments": {}},
    {"tool": "get_problem_status", "arguments": {}},
    {"tool": "read_policy_database", "arguments": {}},
    {"tool": "solve_optimisation_problem", "arguments": {}},
    {"tool": "read_optimal_solution", "arguments": {}},
    {"tool": "render_optimal_solution", "arguments": {}},
]


def ask_ollama_for_next_action(model, context, step_index):
    prompt = (
        "You are an SLM controller for compute provisioning. Return only JSON with keys tool and arguments. "
        "Follow the deterministic workflow unless the context proves the step is invalid. "
        f"Workflow: {json.dumps(WORKFLOW)}\nContext: {context}\nNext step index: {step_index}"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode())
    return json.loads(payload["message"]["content"])


def validated_action(candidate, step_index):
    expected = WORKFLOW[step_index]
    if not isinstance(candidate, dict):
        return expected, "candidate was not a JSON object"
    if candidate.get("tool") != expected["tool"]:
        return expected, f"candidate tool {candidate.get('tool')} replaced with deterministic {expected['tool']}"
    args = candidate.get("arguments", {})
    if args != expected["arguments"]:
        return expected, "candidate arguments replaced with the workflow's arguments"
    return {"tool": expected["tool"], "arguments": args}, "validated"


async def call_mcp_tool(client, name, arguments):
    result = await client.call_tool(name, arguments)
    if hasattr(result, "content"):
        return "\n".join(getattr(item, "text", str(item)) for item in result.content)
    return str(result)


async def run_agent_cycle(model, retries):
    try:
        from fastmcp import Client
    except ImportError as exc:
        raise RuntimeError("fastmcp is required. Install demo/requirements.txt before running the MCP agent.") from exc

    context = []
    server_path = Path(__file__).resolve().parent / "mcp_server.py"
    async with Client(str(server_path)) as client:
        optimisation_called = False
        for step_index in range(len(WORKFLOW)):
            action = None
            validation = ""
            for attempt in range(retries):
                try:
                    candidate = ask_ollama_for_next_action(model, context, step_index)
                except Exception as exc:
                    candidate = WORKFLOW[step_index]
                    validation = f"Ollama unavailable or invalid on attempt {attempt + 1}: {exc}; used deterministic fallback"
                action, validation = validated_action(candidate, step_index)
                if action["tool"] == WORKFLOW[step_index]["tool"]:
                    break
            output = await call_mcp_tool(client, action["tool"], action.get("arguments", {}))
            if action["tool"] == "solve_optimisation_problem":
                optimisation_called = True
            context.append({"action": action, "validation": validation, "output": output})
            print(f"\nTOOL {action['tool']}\nVALIDATION {validation}\n{output}")
        if not optimisation_called:
            print("FAILURE_FLAG: no optimisation tool was called within the retry budget.")
    return context


def workload_size(args, cycle):
    """Viewers and nodes for this cycle: fixed if given, otherwise a rising and falling demand
    curve (slow swell plus a faster ripple and noise) with nodes scaled to match."""
    if args.viewers is not None:
        viewers = args.viewers
    else:
        mid = (args.viewers_min + args.viewers_max) / 2
        amp = (args.viewers_max - args.viewers_min) / 2
        phase = 2 * math.pi * cycle / args.period
        wave = 0.8 * math.sin(phase) + 0.2 * math.sin(3 * phase + 1.0) + random.uniform(-0.05, 0.05)
        viewers = int(round(mid + amp * max(-1.0, min(1.0, wave))))
    nodes = args.nodes if args.nodes is not None else max(4, round(viewers / args.viewers_per_node))
    return viewers, nodes


async def main_async(args):
    initialise_database()
    BUCKETS_TXT.write_text(f"{args.buckets} {args.aggregation}\n")
    if args.policy:
        set_policy(args.policy)
    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== DEMO CYCLE {cycle}: generating workload and ingesting database ===")
        viewers, nodes = workload_size(args, cycle)
        print(f"Workload size: {viewers} viewers, {nodes} compute nodes")
        print(generate_snapshot(viewers, nodes, missing_rate=args.missing_rate, cost_model=args.cost_model))
        await run_agent_cycle(args.model, args.retries)
        if not args.loop:
            break
        await asyncio.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description="Run the standalone Agentic-AI provisioning demo with Ollama + MCP.")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name.")
    parser.add_argument("--policy", default=None, help="Optional priority to write before starting, e.g. 'minimise energy usage'.")
    parser.add_argument("--loop", action="store_true", help="Repeat with a changing workload.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--viewers", type=int, default=None, help="Fixed number of viewers every cycle (disables the demand curve).")
    parser.add_argument("--viewers-min", type=int, default=400, help="Low point of the demand curve.")
    parser.add_argument("--viewers-max", type=int, default=1600, help="High point of the demand curve.")
    parser.add_argument("--period", type=float, default=12, help="Cycles per full rise and fall of demand.")
    parser.add_argument("--viewers-per-node", type=int, default=80, help="Compute nodes scale as viewers / this.")
    parser.add_argument("--nodes", type=int, default=None, help="Fixed number of compute nodes (default: scaled with viewers).")
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS, help="Aggregate viewers into this many weighted buckets before solving (ignored if there are fewer viewers).")
    parser.add_argument("--aggregation", choices=["cost-aware", "spatial"], default="cost-aware", help="cost-aware: bucket by location and cost profile (smaller optimality loss); spatial: bucket by location only (compact, easier to explain).")
    parser.add_argument("--cost-model", choices=["structured", "random"], default="structured", help="structured: node-level delivery costs plus small noise (aggregates well); random: the paper's independent costs.")
    parser.add_argument("--missing-rate", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
