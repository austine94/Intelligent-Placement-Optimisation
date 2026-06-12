# Paper experiments

This directory contains the illustrative benchmark experiment and plotting code for paper figures.

The active paper experiments evaluate six prompt scenarios:

- `focus_cost`: financial-cost priority (`alpha=0.0`, `beta=1.0`).
- `focus_energy`: energy priority (`alpha=0.0`, `beta=0.0`).
- `focus_latency`: latency / user-experience priority (`alpha=1.0`, `beta=0.5`).
- `tradeoff_cost_energy`: cost-energy trade-off with latency not central (`alpha=0.0`, `beta=0.5`).
- `tradeoff_cost_latency`: cost-latency trade-off (`alpha=0.1`, `beta=1.0`).
- `tradeoff_energy_latency`: energy-latency trade-off (`alpha=0.1`, `beta=0.0`).

All paper experiments that ask an Ollama model to choose `solve_allocation` parameters use the shared parameter-selection system prompt and single `solve_allocation` tool definition from `parameter_selection_accuracy.py`. This keeps parameter choices comparable across parameter-accuracy, regret, baseline, illustrative, and scalability experiments.

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

This compares the six scenarios across the temperature sweep `0.0, 0.25, 0.5, 0.75, 1.0` using the shared parameter-selection prompt/tool contract, writes raw JSON results to `experiment_data/regret_temperature_sweep_results.json`, and writes average metric-wise absolute regret percentage tables to:

- `experiment_data/regret_temperature_sweep_average_regret_table.csv`
- `experiment_data/regret_temperature_sweep_average_regret_table.md`

## Regret baseline comparison

Run the regret baseline comparison with:

```bash
python paper_experiments/regret_baseline_comparison.py
```

This compares the six scenarios over 250 generated workloads. For each workload/scenario pair, the script first rewrites the canonical scenario prompt with `llama3.2:latest` to create a fuzzed prompt that preserves the same optimisation requirements. It then evaluates the agentic `aratan/qwen3-4b-tools:latest` baseline at temperature `0.2` using the shared parameter-selection prompt/tool contract, the deterministic keyword classifier, the lightweight semantic classifier, and random parameter selection against that fuzzed prompt. If the agentic baseline fails to produce a usable `solve_allocation` tool call, this experiment only falls back to the keyword classifier so the agentic-baseline row still has a regret value; raw JSON records mark the fallback and keep the original agent failure reason. Raw JSON records also include both the original and fuzzed prompts, plus prompt-generation metadata. The script writes raw JSON results to `experiment_data/regret_baseline_comparison_results.json`, and writes average metric-wise absolute regret percentage tables to:

- `experiment_data/regret_baseline_comparison_average_regret_table.csv`
- `experiment_data/regret_baseline_comparison_average_regret_table.md`

## Regret model sweep

Run the regret model sweep with:

```bash
python paper_experiments/regret_model_sweep.py
```

This compares the six scenarios across the configured locally installed Ollama SLMs `interstellarninja/llama3.1-8b-tools:latest`, `llama3.2:1b`, `llama3.2:latest`, `aratan/qwen3-4b-tools:latest`, and `lukaspetrik/gemma3-tools:4b` at temperature `0.2`. The agent uses the same parameter-selection system prompt and `solve_allocation` tool definition as the model-sweep parameter-accuracy experiment so regret and regret-derived accuracy are measured from the same parameter-selection conditions. It writes raw JSON results to `experiment_data/regret_model_sweep_results.json`, and writes average metric-wise absolute regret percentage tables to:

- `experiment_data/regret_model_sweep_average_regret_table.csv`
- `experiment_data/regret_model_sweep_average_regret_table.md`

## Regret model-sweep parameter-selection accuracy

After running the regret model sweep, compute parameter-selection accuracy from the same raw records with:

```bash
python paper_experiments/regret_model_sweep_parameter_selection_accuracy.py
```

This post-processing step reads `experiment_data/regret_model_sweep_results.json`, scores the `agent_tool_args` selected during that regret experiment using the same `±0.05` alpha/beta tolerance and combined score as the standalone parameter-accuracy sweeps, and writes aligned model-sweep accuracy outputs to:

- `experiment_data/regret_model_sweep_parameter_selection_accuracy_results.json`
- `experiment_data/regret_model_sweep_parameter_selection_accuracy_table.csv`
- `experiment_data/regret_model_sweep_parameter_selection_accuracy_table.md`


## Varying-prompt regret model sweep

Run the varying-prompt regret model sweep with:

```bash
python paper_experiments/varying_prompt_regret_model_sweep.py
```

This follows the regret model sweep setup at temperature `0.2`, but for each generated workload/scenario pair it first rewrites the canonical prompt with `llama3.2:latest` using the same prompt-generation mechanism as the regret baseline comparison. The generated prompt variant is shared across all configured models for that workload/scenario pair, so each model is evaluated against the same wording. Raw JSON records include both the original and generated prompts, plus prompt-generation metadata. The script writes raw JSON results to `experiment_data/varying_prompt_regret_model_sweep_results.json`, and writes average metric-wise absolute regret percentage tables to:

- `experiment_data/varying_prompt_regret_model_sweep_average_regret_table.csv`
- `experiment_data/varying_prompt_regret_model_sweep_average_regret_table.md`

## Varying-prompt regret model-sweep parameter-selection accuracy

After running the varying-prompt regret model sweep, compute parameter-selection accuracy from the same raw records with:

```bash
python paper_experiments/varying_prompt_regret_model_sweep_parameter_selection_accuracy.py
```

This post-processing step reads `experiment_data/varying_prompt_regret_model_sweep_results.json`, scores the `agent_tool_args` selected during that regret experiment using the same `±0.05` alpha/beta tolerance and combined score as the standalone parameter-accuracy sweeps, and writes aligned varying-prompt model-sweep accuracy outputs to:

- `experiment_data/varying_prompt_regret_model_sweep_parameter_selection_accuracy_results.json`
- `experiment_data/varying_prompt_regret_model_sweep_parameter_selection_accuracy_table.csv`
- `experiment_data/varying_prompt_regret_model_sweep_parameter_selection_accuracy_table.md`

## Parameter accuracy temperature sweep

Run the parameter-selection accuracy temperature sweep with:

```bash
python paper_experiments/parameter_accuracy_temperature_sweep.py
```

This compares the six scenarios across the temperature sweep `0.0, 0.25, 0.5, 0.75, 1.0`. A selected alpha or beta is counted as correct when it is within `±0.05` of the scenario target. Each repetition receives a combined alpha/beta score: `1.0` when both are correct, `0.5` when exactly one is correct, and `0.0` when neither is correct. The script writes raw JSON results to `experiment_data/parameter_accuracy_temperature_sweep_results.json`, and writes proportion-correct tables to:

- `experiment_data/parameter_accuracy_temperature_sweep_accuracy_table.csv`
- `experiment_data/parameter_accuracy_temperature_sweep_accuracy_table.md`

After running the sweep, generate a temperature-on-x-axis line figure with one line per scenario with:

```bash
python paper_experiments/plot_parameter_accuracy_temperature_sweep.py
```

The plotting script reads `experiment_data/parameter_accuracy_temperature_sweep_results.json` without rerunning Ollama models and writes:

- `paper_plots/parameter_accuracy_temperature_sweep_by_scenario.pdf`

## Parameter accuracy model sweep

Run the parameter-selection accuracy model sweep with:

```bash
python paper_experiments/parameter_accuracy_model_sweep.py
```

This compares the six scenarios across the configured locally installed Ollama SLMs `interstellarninja/llama3.1-8b-tools:latest`, `llama3.2:1b`, `llama3.2:latest`, `aratan/qwen3-4b-tools:latest`, and `lukaspetrik/gemma3-tools:4b` at temperature `0.2`. It uses the same `±0.05` parameter tolerance and combined alpha/beta repetition score as the temperature sweep. The script writes raw JSON results to `experiment_data/parameter_accuracy_model_sweep_results.json`, and writes proportion-correct tables to:

- `experiment_data/parameter_accuracy_model_sweep_accuracy_table.csv`
- `experiment_data/parameter_accuracy_model_sweep_accuracy_table.md`

## Parameter accuracy standard-deviation tables

After running both standalone parameter-selection accuracy sweeps, compute standard-deviation tables from the raw records with:

```bash
python paper_experiments/parameter_accuracy_selection_std_tables.py
```

This post-processing script reads `experiment_data/parameter_accuracy_model_sweep_results.json` and `experiment_data/parameter_accuracy_temperature_sweep_results.json` without rerunning any Ollama models. For each scenario and sweep column, it writes the population standard deviation of the combined alpha/beta `parameter_selection_score` to:

- `experiment_data/parameter_accuracy_model_sweep_parameter_selection_standard_deviation_table.csv`
- `experiment_data/parameter_accuracy_model_sweep_parameter_selection_standard_deviation_table.md`
- `experiment_data/parameter_accuracy_temperature_sweep_parameter_selection_standard_deviation_table.csv`
- `experiment_data/parameter_accuracy_temperature_sweep_parameter_selection_standard_deviation_table.md`

After generating the standard-deviation tables, create a temperature-on-x-axis standard-deviation figure with one line per scenario with:

```bash
python paper_experiments/plot_parameter_accuracy_temperature_sweep_std.py
```

The plotting script reads `experiment_data/parameter_accuracy_temperature_sweep_parameter_selection_standard_deviation_table.csv` and writes:

- `paper_plots/parameter_accuracy_temperature_sweep_standard_deviation_by_scenario.pdf`

To combine mean selection accuracy and standard deviation into one dual-axis figure, run:

```bash
python paper_experiments/plot_parameter_accuracy_temperature_sweep_combined.py
```

The combined plotting script reads both the temperature-sweep raw JSON and standard-deviation CSV. It writes a figure with mean accuracy on the left y-axis, standard deviation on the right y-axis, one colour per scenario, solid lines for means, and dashed lines for standard deviations:

- `paper_plots/parameter_accuracy_temperature_sweep_combined_by_scenario.pdf`

## Scalability experiment

Run the cost-latency trade-off scalability experiment with:

```bash
python paper_experiments/scalability_experiment.py
```

This generates workloads with `50, 100, ..., 1000` consumers using `scripts/scalability_data_generator.py`, with 30 repetitions per workload size. Each workload uses 5 datacentres with capacity 1100 each so the 1000-consumer instances are feasible. For each generated workload, the script evaluates the `tradeoff_cost_latency` scenario across the configured locally installed Ollama SLMs `interstellarninja/llama3.1-8b-tools:latest`, `llama3.2:1b`, `llama3.2:latest`, `aratan/qwen3-4b-tools:latest`, and `lukaspetrik/gemma3-tools:4b` at temperature `0.2` using the shared parameter-selection prompt/tool contract. It records metric-wise absolute regret percentage against the cost-latency target, all top-level Ollama count fields, total token count, agent end-to-end time, and total case time. Raw JSON results are written to `experiment_data/scalability_experiment_results.json`, and average metric tables are written to:

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
