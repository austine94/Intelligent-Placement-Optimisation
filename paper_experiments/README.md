# Paper experiments

This directory contains the illustrative benchmark experiment and plotting code for paper figures.

The active paper experiments evaluate six prompt scenarios:

- `focus_cost`: financial-cost priority (`alpha=0.0`, `beta=1.0`).
- `focus_energy`: energy priority (`alpha=0.0`, `beta=0.0`).
- `focus_latency`: latency / user-experience priority (`alpha=1.0`, `beta=0.5`).
- `tradeoff_cost_energy`: cost-energy trade-off with latency not central (`alpha=0.0`, `beta=0.5`).
- `tradeoff_cost_latency`: cost-latency trade-off (`alpha=0.1`, `beta=1.0`).
- `tradeoff_energy_latency`: energy-latency trade-off (`alpha=0.1`, `beta=0.0`).

## Illustrative experiment

`illustrative_experiment.py` is a copy of the benchmark agent experiment configured to save its JSON output as `experiment_data/illustrative_experiment_results.json`.
`plot_illustrative_experiment.py` reads `experiment_data/illustrative_experiment_results.json`, then writes PDF plot outputs to `paper_plots/`. The scenario panels plot normalised metric means with standard-deviation error bars so cost, energy, and latency can share a single axis.

Run the experiment first:

```bash
python paper_experiments/illustrative_experiment.py
```

This writes the experiment JSON output to:

- `experiment_data/illustrative_experiment_results.json`

Then generate the plot outputs:

```bash
python paper_experiments/plot_illustrative_experiment.py
```

The plotting script creates one scenario PDF output per benchmark prompt:

- `paper_plots/illustrative_experiment_focus_cost.pdf`
- `paper_plots/illustrative_experiment_focus_energy.pdf`
- `paper_plots/illustrative_experiment_focus_latency.pdf`
- `paper_plots/illustrative_experiment_tradeoff_cost_energy.pdf`
- `paper_plots/illustrative_experiment_tradeoff_cost_latency.pdf`
- `paper_plots/illustrative_experiment_tradeoff_energy_latency.pdf`

It also creates the combined 2 x 3 PDF output:

- `paper_plots/illustrative_experiment_all.pdf`

The JSON and PDF files are generated outputs, not source scripts, so they are ignored by git and can be regenerated whenever experiment results change.

## Regret temperature sweep

Run the regret temperature sweep with:

```bash
python paper_experiments/regret_temperature_sweep.py
```

This compares the six scenarios across the temperature sweep `0.0, 0.25, 0.5, 0.75, 1.0`, writes raw JSON results to `experiment_data/regret_temperature_sweep_results.json`, and writes average objective-regret percentage tables to:

- `experiment_data/regret_temperature_sweep_average_regret_table.csv`
- `experiment_data/regret_temperature_sweep_average_regret_table.md`

## Regret model sweep

Run the regret model sweep with:

```bash
python paper_experiments/regret_model_sweep.py
```

This compares the six scenarios across the configured locally installed Ollama SLMs `interstellarninja/llama3.1-8b-tools:latest`, `llama3.2:1b`, `llama3:8b`, `aratan/qwen3-4b-tools:latest`, and `lukaspetrik/gemma3-tools:4b` at temperature `0.0`. It writes raw JSON results to `experiment_data/regret_model_sweep_results.json`, and writes average objective-regret percentage tables to:

- `experiment_data/regret_model_sweep_average_regret_table.csv`
- `experiment_data/regret_model_sweep_average_regret_table.md`

## Parameter accuracy temperature sweep

Run the parameter-selection accuracy temperature sweep with:

```bash
python paper_experiments/parameter_accuracy_temperature_sweep.py
```

This compares the six scenarios across the temperature sweep `0.0, 0.25, 0.5, 0.75, 1.0`. A selected alpha or beta is counted as correct when it is within `±0.05` of the scenario target. Each repetition receives a combined alpha/beta score: `1.0` when both are correct, `0.5` when exactly one is correct, and `0.0` when neither is correct. The script writes raw JSON results to `experiment_data/parameter_accuracy_temperature_sweep_results.json`, and writes proportion-correct tables to:

- `experiment_data/parameter_accuracy_temperature_sweep_accuracy_table.csv`
- `experiment_data/parameter_accuracy_temperature_sweep_accuracy_table.md`

## Parameter accuracy model sweep

Run the parameter-selection accuracy model sweep with:

```bash
python paper_experiments/parameter_accuracy_model_sweep.py
```

This compares the six scenarios across the configured locally installed Ollama SLMs `interstellarninja/llama3.1-8b-tools:latest`, `llama3.2:1b`, `llama3.2:latest`, `aratan/qwen3-4b-tools:latest`, and `lukaspetrik/gemma3-tools:4b` at temperature `0.0`. It uses the same `±0.05` parameter tolerance and combined alpha/beta repetition score as the temperature sweep. The script writes raw JSON results to `experiment_data/parameter_accuracy_model_sweep_results.json`, and writes proportion-correct tables to:

- `experiment_data/parameter_accuracy_model_sweep_accuracy_table.csv`
- `experiment_data/parameter_accuracy_model_sweep_accuracy_table.md`

## Scalability experiment

Run the cost-latency trade-off scalability experiment with:

```bash
python paper_experiments/scalability_experiment.py
```

This generates workloads with `50, 100, ..., 1000` consumers using `scripts/scalability_data_generator.py`, with 30 repetitions per workload size. Each workload uses 5 datacentres with capacity 1100 each so the 1000-consumer instances are feasible. For each generated workload, the script evaluates the `tradeoff_cost_latency` scenario across the configured locally installed Ollama SLMs `interstellarninja/llama3.1-8b-tools:latest`, `llama3.2:1b`, `llama3:8b`, `aratan/qwen3-4b-tools:latest`, and `lukaspetrik/gemma3-tools:4b` at temperature `0.0`. It records objective-regret percentage against the cost-latency target, all top-level Ollama count fields, total token count, agent end-to-end time, and total case time. Raw JSON results are written to `experiment_data/scalability_experiment_results.json`, and average metric tables are written to:

- `experiment_data/scalability_average_metrics_table.csv`
- `experiment_data/scalability_average_metrics_table.md`

Then generate the scalability plot outputs with:

```bash
python paper_experiments/plot_scalability_experiment.py
```

The plotting script creates one PDF line plot per aggregate metric:

- `paper_plots/scalability_average_regret.pdf`
- `paper_plots/scalability_average_tokens.pdf`
- `paper_plots/scalability_average_agent_time.pdf`
- `paper_plots/scalability_average_total_case_time.pdf`
